from typing import Dict, Any, Union, List
from openai.types.create_embedding_response import CreateEmbeddingResponse
from .base import OpenAICapability

# Map OpenAI embedding models to our models
MODEL_MAPPINGS = {
    'text-embedding-ada-002': '@cf/baai/bge-base-en-v1.5',  # 768 dimensions
    'text-embedding-3-small': '@cf/baai/bge-base-en-v1.5',  # 768 dimensions
}

class Embeddings(OpenAICapability):
    """Embeddings capability using OpenAI-compatible endpoints"""

    def create(
        self,
        input: Union[str, List[str], List[int], List[List[int]]],
        model: str = "text-embedding-ada-002",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create embeddings for the given input
        
        Args:
            input: Text or array of text/tokens to embed
            model: Model to use for embeddings
            **kwargs: Additional parameters
            
        Returns:
            Response containing the generated embeddings
        """
        # Map OpenAI model to our model
        mapped_model = MODEL_MAPPINGS.get(model, model)
        
        response = self._openai.embeddings.create(
            input=input,
            model=mapped_model,
            **kwargs
        )
        
        # Convert OpenAI response to compatible format
        return {
            "data": [
                {"embedding": embedding.embedding} 
                for embedding in response.data
            ],
            "model": response.model,
            "usage": response.usage.dict() if response.usage else {}
        }
