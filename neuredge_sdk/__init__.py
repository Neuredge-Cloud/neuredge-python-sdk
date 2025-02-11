from typing import Optional
from .client import NeuredgeClient
from .types import NeuredgeError

class Neuredge:
    """
    Main class for interacting with the Neuredge API
    
    Example:
        ```python
        from neuredge_sdk import Neuredge
        
        client = Neuredge(
            api_key="your-api-key",
            max_retries=3,
            retry_delay=1.0
        )
        
        # Use core capabilities
        summary = client.text.summarize("...")
        
        # Use OpenAI-compatible endpoints
        completion = client.openai.chat.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.neuredge.dev",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self._client = NeuredgeClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Core capabilities
        self.text = self._client.text
        self.image = self._client.image
        self.vector = self._client.vector
        
        # OpenAI-compatible endpoints
        self.openai = self._client.openai

    def close(self):
        """Close the client and cleanup resources"""
        self._client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Export types and errors
__all__ = ['Neuredge', 'NeuredgeError']
