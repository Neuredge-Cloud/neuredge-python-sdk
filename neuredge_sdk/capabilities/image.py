from typing import Optional, Dict, Any, List, Union
from ..types import ImageGenerationOptions
from .base import BaseCapability

class ImageCapabilities(BaseCapability):
    @property
    def base_path(self) -> str:
        return '/image'  # Image endpoints use /image prefix

    def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description of the desired image
            options: Image generation options including mode, dimensions, etc.
            
        Returns:
            The generated image as bytes
        """
        options = options or {}
        request_data = {
            'prompt': prompt,
            'model': options.get('model', "@cf/stabilityai/stable-diffusion-xl-base-1.0"),
            'width': options.get('width', 1024),
            'height': options.get('height', 1024),
            'style': options.get('style')
        }

        response = self._client.post(
            self.endpoint('/generate'),
            request_data,
            binary_response=True  # Indicate we expect binary data
        )

        # Convert binary response to base64 string
        if isinstance(response, bytes):
            import base64
            image_b64 = base64.b64encode(response).decode('utf-8')
            return {'images': [f"data:image/jpeg;base64,{image_b64}"]}
        
        # Handle JSON response format
        if isinstance(response, dict):
            return {'images': response.get('images', [])}
        
        # Handle direct string response
        if isinstance(response, str):
            return {'images': [response]}

        raise ValueError(f"Unexpected response type: {type(response)}")

    def generate_fast(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate an image quickly with potentially lower quality
        
        Args:
            prompt: Text description of the desired image
            options: Image generation options (excluding mode)
            
        Returns:
            The generated image as bytes
        """
        options = options or {}
        options['mode'] = 'fast'
        return self.generate(prompt, options)

    def generate_standard(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate a high-quality image with standard processing time
        
        Args:
            prompt: Text description of the desired image
            options: Image generation options (excluding mode)
            
        Returns:
            The generated image as bytes
        """
        options = options or {}
        options['mode'] = 'standard'
        return self.generate(prompt, options)
