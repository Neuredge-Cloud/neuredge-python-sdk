from typing import Optional
from ..types import (
    LanguageCode,
    SentimentResult,
    ApiResponse,
    SummaryResult,
    TranslationResult
)
from .base import BaseCapability

class TextCapabilities(BaseCapability):
    @property
    def base_path(self) -> str:
        return ''  # Text endpoints don't use a prefix

    def summarize(self, text: str) -> str:
        """
        Generate a summary of the provided text
        
        Args:
            text: The text to summarize
            
        Returns:
            A concise summary of the input text
        """
        response = self._client.post(
            self.endpoint('/summarize'),
            {'text': text}
        )
        return response['result']['summary']

    def translate(
        self,
        text: str,
        target_lang: LanguageCode,
        source_lang: Optional[LanguageCode] = None
    ) -> str:
        """
        Translate text from one language to another
        
        Args:
            text: The text to translate
            target_lang: The target language code
            source_lang: Optional source language code (auto-detected if not provided)
            
        Returns:
            The translated text
        """
        request_data = {
            'text': text,
            'target_lang': target_lang
        }
        if source_lang:
            request_data['source_lang'] = source_lang

        response = self._client.post(
            self.endpoint('/translate'),
            request_data
        )
        return response['result']['translation']

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of the provided text
        
        Args:
            text: The text to analyze
            
        Returns:
            Sentiment analysis result with confidence score
        """
        response = self._client.post(
            self.endpoint('/sentiment'),
            {'text': text}
        )
        return response['result']
