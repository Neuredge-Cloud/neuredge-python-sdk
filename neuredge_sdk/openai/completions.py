from typing import Dict, Any, Iterator, Optional, Union
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from .base import OpenAICapability

# Map OpenAI models to our supported models
MODEL_MAPPINGS = {
    'gpt-3.5-turbo': '@cf/meta/llama-3.1-8b-instruct',  # Default model
    'gpt-4': '@cf/meta/llama-3.1-70b-instruct',  # More capable model
}

class ChatCompletions(OpenAICapability):
    """Chat completions capability using OpenAI-compatible endpoints"""

    def create(
        self,
        messages: list[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a chat completion
        
        Args:
            messages: List of chat messages in the conversation
            model: Model to use for completion
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        # Map OpenAI model to our supported model
        mapped_model = MODEL_MAPPINGS.get(model, model)
        
        response = self._openai.chat.completions.create(
            messages=messages,
            model=mapped_model,
            stream=stream,
            **kwargs
        )

        if not stream:
            return self._format_completion(response)
        return self._stream_completion(response)

    def _format_completion(self, response: ChatCompletion) -> Dict[str, Any]:
        """Format a regular completion response"""
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                },
                "finish_reason": choice.finish_reason
            } for choice in response.choices],
            "usage": response.usage.dict() if response.usage else {}
        }

    def _stream_completion(self, response: Iterator[ChatCompletionChunk]) -> Iterator[Dict[str, Any]]:
        """Format streaming completion responses"""
        for chunk in response:
            if not chunk.choices:
                continue
            yield {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": chunk.created,
                "model": chunk.model,
                "choices": [{
                    "index": choice.index,
                    "delta": {
                        "role": choice.delta.role if choice.delta.role else None,
                        "content": choice.delta.content if choice.delta.content else None
                    },
                    "finish_reason": choice.finish_reason
                } for choice in chunk.choices]
            }
