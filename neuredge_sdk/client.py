from typing import Optional, Dict, Any, TypeVar, Generic
import requests
from dataclasses import dataclass
import time

from .types import ClientConfig, NeuredgeError, ApiResponse

T = TypeVar('T')

class NeuredgeClient:
    """Main client for interacting with the Neuredge API"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.neuredge.dev",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })

        # Initialize capabilities
        from .capabilities.text import TextCapabilities
        from .capabilities.image import ImageCapabilities
        from .capabilities.vector import VectorStoreCapabilities
        from .openai.index import OpenAINamespace

        self.text = TextCapabilities(self)
        self.image = ImageCapabilities(self)
        self.vector = VectorStoreCapabilities(self)
        self.openai = OpenAINamespace(self)

    def get_api_key(self) -> str:
        """Get the API key used by this client"""
        return self._api_key

    def get_base_url(self) -> str:
        """Get the base URL used by this client"""
        return self._base_url

    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and errors"""
        try:
            json_response = response.json()
        except ValueError:
            if response.status_code >= 400:
                # Map status codes to appropriate errors
                if response.status_code == 401:
                    raise NeuredgeError(
                        message='Invalid API key',
                        code='AUTHENTICATION_ERROR',
                        status_code=401
                    )
                raise NeuredgeError(
                    message=f"HTTP {response.status_code} error",
                    code='REQUEST_FAILED',
                    status_code=response.status_code
                )
            return response.content

        if not response.ok:
            error_data = json_response.get('error', {})
            if isinstance(error_data, dict):
                raise NeuredgeError(
                    message=error_data.get('message', 'Unknown error'),
                    code=error_data.get('type', 'UNKNOWN_ERROR').upper(),
                    status_code=response.status_code,
                    details=error_data
                )
            raise NeuredgeError(
                message=str(error_data),
                code='REQUEST_FAILED',
                status_code=response.status_code
            )

        return json_response

    def _retry_request(self, method: str, *args, **kwargs) -> Any:
        """Make a request with retries"""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = getattr(self._session, method)(*args, **kwargs)
                return self._handle_response(response)
            except requests.RequestException as e:
                last_error = NeuredgeError(
                    f"Network error: {str(e)}",
                    'NETWORK_ERROR'
                )
            except NeuredgeError as e:
                # Don't retry authentication or quota errors
                if e.code in ['AUTHENTICATION_ERROR', 'QUOTA_EXCEEDED']:
                    raise
                last_error = e
            
            # Wait before retrying
            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (2 ** attempt))
        
        raise last_error

    def post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        binary_response: bool = False
    ) -> Any:
        """
        Make a POST request to the API
        
        Args:
            endpoint: API endpoint path
            data: Request body data
            binary_response: Whether to expect a binary response
            
        Returns:
            Parsed response data or binary data for images
        """
        url = f"{self._base_url}{endpoint}"
        return self._retry_request('post', url, json=data)

    def get(self, endpoint: str) -> Any:
        """
        Make a GET request to the API
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Parsed response data
        """
        url = f"{self._base_url}{endpoint}"
        return self._retry_request('get', url)

    def delete(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a DELETE request to the API
        
        Args:
            endpoint: API endpoint path
            data: Optional request body data
            
        Returns:
            Parsed response data
        """
        url = f"{self._base_url}{endpoint}"
        return self._retry_request('delete', url, json=data if data else None)

    def close(self):
        """Close the requests session"""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

class Client(NeuredgeClient):
    """Compatibility class that inherits from NeuredgeClient"""
    pass  # All functionality is inherited from NeuredgeClient
