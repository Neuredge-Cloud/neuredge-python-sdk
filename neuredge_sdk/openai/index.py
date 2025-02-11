from typing import Any
from .base import OpenAICapability
from .completions import ChatCompletions
from .embeddings import Embeddings

class OpenAINamespace(OpenAICapability):
    """
    OpenAI-compatible API namespace containing chat completions and embeddings
    Mimics the OpenAI SDK structure for familiarity
    """

    def __init__(self, client: Any):
        super().__init__(client)
        self._chat = ChatCompletions(client)
        self.embeddings = Embeddings(client)

    @property
    def base_path(self) -> str:
        return '/v1'  # OpenAI endpoints use v1 prefix

    @property
    def chat(self) -> ChatCompletions:
        """Access to chat completion endpoints"""
        return self._chat
