from typing import Any, Optional
from openai import OpenAI
from ..capabilities.base import BaseCapability
from ..client import Client

class OpenAICapability(BaseCapability):
    """Base class for OpenAI-compatible capabilities"""

    def __init__(self, client: Client):
        self._client = client
        self._openai = OpenAI(
            api_key=client.get_api_key(),
            base_url=client.get_base_url() + self.base_path
        )

    @property
    def base_path(self) -> str:
        return '/v1'  # OpenAI endpoints use v1 prefix

    def endpoint(self, path: str) -> str:
        return f"{self.base_path}{path}"

    def _format_openai_error(self, error_data: dict) -> str:
        """Format OpenAI-style error messages"""
        if 'error' in error_data:
            error = error_data['error']
            if isinstance(error, dict):
                return error.get('message', str(error))
            return str(error)
        return str(error_data)
