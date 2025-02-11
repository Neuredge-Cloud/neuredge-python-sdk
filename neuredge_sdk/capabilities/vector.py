from typing import List, Optional, Union, Dict, Any
import time

from ..types import (
    VectorIndex,
    Vector,
    VectorOptions,
    VectorIndexResponse,
    SearchVectorMatch,
    AddVectorsResult,
    NeuredgeError
)
from .base import BaseCapability

class VectorStoreCapabilities(BaseCapability):
    @property
    def base_path(self) -> str:
        return '/v1'  # Vector store endpoints use v1 prefix

    def create_index(self, config: VectorIndex) -> None:
        """
        Create a new vector index
        
        Args:
            config: Vector index configuration
        """
        try:
            self._client.post(self.endpoint('/indexes'), config)
            # Wait briefly to allow index creation to propagate
            time.sleep(1)
        except Exception as e:
            raise e

    def list_indexes(self) -> List[VectorIndex]:
        """
        List all vector indexes
        
        Returns:
            List of vector index configurations
        """
        response = self._client.get(self.endpoint('/indexes'))
        
        # Handle direct response structure without result wrapper
        indexes_data = response.get('indexes', [])
        
        return [
            {
                'name': index['name'],
                'dimension': index['dimension'],
                'metric': 'cosine',
                'vector_count': index.get('vector_count', 0)
            }
            for index in indexes_data
        ]

    def get_index(self, name: str) -> Optional[VectorIndex]:
        """
        Get details of a specific index
        
        Args:
            name: Name of the index
            
        Returns:
            Vector index configuration if found, None otherwise
        """
        try:
            response = self._client.get(self.endpoint(f'/indexes/{name}'))
            
            if not response:
                return None

            # Ensure we have required properties
            if 'name' not in response or 'dimension' not in response:
                print('Warning: Invalid index response format:', response)
                return None

            return {
                'name': response['name'],
                'dimension': response['dimension'],
                'metric': 'cosine',
                'vector_count': response.get('vector_count', 0)
            }
        except NeuredgeError as e:
            if e.status_code == 404:
                return None
            raise e

    def delete_index(self, name: str) -> None:
        """
        Delete a vector index
        
        Args:
            name: Name of the index to delete
        """
        try:
            self._client.delete(self.endpoint(f'/indexes/{name}'))
        except NeuredgeError as e:
            # Ignore 404 errors during cleanup
            if e.status_code == 404:
                return
            raise e

    def add_vectors(
        self, 
        index_name: str, 
        vectors: List[Vector],
        options: Optional[Dict[str, Any]] = None
    ) -> AddVectorsResult:
        """
        Store vectors in an index
        
        Args:
            index_name: Name of the index
            vectors: Array of vectors to store
            options: Vector operation options
            
        Returns:
            Result containing number of vectors inserted and their IDs
        """
        options = options or {}
        consistency = options.get('consistency', {})
        
        # Get current count if consistency mode is enabled
        before_count = 0
        if consistency.get('enabled'):
            index = self.get_index(index_name)
            if not index:
                raise NeuredgeError(
                    f"Index {index_name} not found",
                    'INDEX_NOT_FOUND',
                    404
                )
            before_count = index.get('vector_count', 0)

        # Store vectors
        response = self._client.post(
            self.endpoint(f'/indexes/{index_name}/vectors'),
            {'vectors': vectors}
        )

        # Ensure response has required properties
        if not response or 'inserted' not in response:
            raise NeuredgeError(
                'Invalid response from add vectors',
                'INVALID_RESPONSE',
                500
            )

        # Wait for vectors to be indexed if consistency mode is enabled
        if consistency.get('enabled'):
            expected_count = before_count + len(vectors)
            max_retries = consistency.get('max_retries', 5)
            retry_delay = consistency.get('retry_delay', 3000) / 1000  # Convert to seconds

            for attempt in range(max_retries):
                index = self.get_index(index_name)
                if index and index.get('vector_count') == expected_count:
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return {
            'inserted': response['inserted'],
            'ids': response.get('ids', [])
        }

    def delete_vectors(
        self,
        index_name: str,
        ids: List[Union[str, int]]
    ) -> None:
        """
        Delete vectors from an index
        
        Args:
            index_name: Name of the index
            ids: Array of vector IDs to delete
        """
        self._client.delete(
            self.endpoint(f'/indexes/{index_name}/vectors'),
            {'ids': ids}
        )

    def search_vector(
        self,
        index_name: str,
        vector: List[float],
        options: Optional[Dict[str, Any]] = None
    ) -> List[SearchVectorMatch]:
        """
        Search for similar vectors
        
        Args:
            index_name: Name of the index
            vector: Query vector
            options: Search and consistency options
            
        Returns:
            List of matched vectors with similarity scores
        """
        options = options or {}
        consistency = options.get('consistency', {})
        max_retries = consistency.get('max_retries', 1)
        retry_delay = consistency.get('retry_delay', 0) / 1000  # Convert to seconds

        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    self.endpoint(f'/indexes/{index_name}/search'),
                    {
                        'vector': vector,
                        'limit': options.get('top_k', 10),
                    }
                )
                
                # If we have results, return them immediately
                if response.get('results') and len(response['results']) > 0:
                    return response['results']
                
                # Only retry if we have no results and consistency is enabled
                if not consistency.get('enabled'):
                    return response.get('results', [])

                if attempt < max_retries - 1:
                    print(f"Search attempt {attempt + 1}/{max_retries}: No results, retrying...")
                    time.sleep(retry_delay)

            except Exception as e:
                print(f"Search attempt {attempt + 1} failed:", e)
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)

        return []
