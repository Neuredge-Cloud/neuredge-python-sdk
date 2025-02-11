from typing import Optional, Dict, Any, List, Union
from ..types import ImageGenerationOptions
from .base import BaseCapability
import base64

class ImageCapabilities(BaseCapability):
    """Image generation capabilities"""

    @property
    def base_path(self) -> str:  # Changed from basePath to base_path to match abstract method
        return '/image'

    def _convert_to_bytes(self, response: Any) -> bytes:
        """Convert API response to bytes"""
        # Handle direct binary response
        if isinstance(response, bytes):
            return response
            
        # Handle response with images array
        if isinstance(response, dict) and 'images' in response:
            image_str = response['images'][0]
            if image_str.startswith('data:image/'):
                base64_data = image_str.split(',')[1]
            else:
                base64_data = image_str
            return base64.b64decode(base64_data)
        
        # Log the actual response format for debugging
        if isinstance(response, dict):
            print("Response keys:", response.keys())
        print("Response type:", type(response))
        raise ValueError("Unexpected response format")

    def generate(self, prompt: str, options: Optional[Dict] = None) -> bytes:
        """Generate image and return as bytes (like JS Blob)"""
        response = self._client.post(
            f"{self.base_path}/generate", 
            {
                "prompt": prompt,
                **(options or {})
            },
            binary_response=True  # Add flag for binary response
        )
        return self._convert_to_bytes(response)

    def generate_fast(self, prompt: str, options: Optional[Dict] = None) -> bytes:
        """Quick generation, returns bytes"""
        return self.generate(prompt, {
            **(options or {}),
            "mode": "fast"
        })

    def generate_standard(self, prompt: str, options: Optional[Dict] = None) -> bytes:
        """Standard generation, returns bytes"""
        return self.generate(prompt, {
            **(options or {}),
            "mode": "standard"
        })
