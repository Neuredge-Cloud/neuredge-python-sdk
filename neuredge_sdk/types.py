from typing import Dict, List, Optional, Union, TypeVar, Generic, Any
from typing_extensions import NotRequired, TypedDict, Literal
from enum import Enum
from dataclasses import dataclass

T = TypeVar('T')

class ErrorCode(str, Enum):
    """API Error Codes"""
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'
    NETWORK_ERROR = 'NETWORK_ERROR'
    REQUEST_FAILED = 'REQUEST_FAILED'
    QUOTA_EXCEEDED = 'QUOTA_EXCEEDED'
    INDEX_NOT_FOUND = 'INDEX_NOT_FOUND'
    INDEX_EXISTS = 'INDEX_EXISTS'
    INVALID_REQUEST = 'INVALID_REQUEST'
    INVALID_RESPONSE = 'INVALID_RESPONSE'

class NeuredgeError(Exception):
    """Standardized error handling for Neuredge SDK"""
    
    def __init__(
        self,
        message: str,
        code: str = 'UNKNOWN_ERROR',
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    @classmethod
    def from_openai_error(cls, error: Dict[str, Any]) -> 'NeuredgeError':
        """Convert OpenAI-style error to NeuredgeError"""
        return cls(
            message=error.get('error', {}).get('message', 'Unknown error'),
            code=error.get('error', {}).get('type', 'UNKNOWN_ERROR').upper(),
            status_code=error.get('status', 500),
            details=error.get('error')
        )

    @classmethod
    def from_http_error(cls, status_code: int, message: str) -> 'NeuredgeError':
        """Convert HTTP error to NeuredgeError"""
        if status_code == 401:
            return cls(
                message='Invalid API key',
                code='AUTHENTICATION_ERROR',
                status_code=401
            )
        return cls(
            message=message,
            code='HTTP_ERROR',
            status_code=status_code
        )

    def __str__(self) -> str:
        """Consistent error message format"""
        return f"Error code: {self.status_code} - {self.message}"

class ClientConfig(TypedDict):
    """API Configuration options"""
    api_key: str
    base_url: NotRequired[str]  # Optional with default "https://api.neuredge.dev"
    max_retries: NotRequired[int]  # Optional with default 3
    retry_delay: NotRequired[float]  # Optional with default 1.0

class ApiMetadata(TypedDict):
    compression_ratio: float
    original_length: int
    summary_length: int

class ApiUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ApiQuota(TypedDict):
    limit: int
    used: int
    remaining: int
    current_request: int

class ApiResponseData(TypedDict, Generic[T]):
    result: T
    usage: ApiUsage
    quota: ApiQuota

class ApiErrorData(TypedDict):
    code: str
    message: str
    details: NotRequired[dict]

class ApiResponse(TypedDict, Generic[T]):
    """Common response format for all API endpoints"""
    success: bool
    data: NotRequired[ApiResponseData[T]]
    error: NotRequired[ApiErrorData]

# Vector Store Types
class VectorMetric(str, Enum):
    COSINE = 'cosine'
    EUCLIDEAN = 'euclidean'
    DOT = 'dot'

class VectorIndex(TypedDict):
    """Vector Store Index Configuration"""
    name: str
    dimension: int
    metric: VectorMetric
    vector_count: NotRequired[int]

class Vector(TypedDict):
    """Vector with ID for storage"""
    id: Union[str, int]
    values: List[float]

class SearchVectorMatch(TypedDict):
    """Search result for vector similarity search"""
    id: Union[str, int]
    score: float
    vector: NotRequired[List[float]]

class ConsistencyOptions(TypedDict):
    enabled: bool
    max_retries: NotRequired[int]  # default 5
    retry_delay: NotRequired[int]  # default 3000ms

class VectorOptions(TypedDict):
    """Options for vector operations"""
    top_k: NotRequired[int]  # default 10
    consistency: NotRequired[ConsistencyOptions]

# API Response Types
class VectorIndexResponse(TypedDict):
    id: int
    user_id: int
    name: str
    dimension: int
    vector_count: int
    created_at: str
    deleted: int
    deleted_at: Optional[str]

class ListVectorIndexesResult(TypedDict):
    indexes: List[VectorIndexResponse]

class AddVectorsResult(TypedDict):
    inserted: int
    ids: List[str]

class SearchVectorResult(TypedDict):
    results: List[SearchVectorMatch]

# Language and Text Types
class LanguageCode(str, Enum):
    EN = 'en'  # English
    ES = 'es'  # Spanish
    FR = 'fr'  # French
    DE = 'de'  # German
    IT = 'it'  # Italian
    PT = 'pt'  # Portuguese
    RU = 'ru'  # Russian
    ZH = 'zh'  # Chinese
    JA = 'ja'  # Japanese
    KO = 'ko'  # Korean

class SentimentType(str, Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'

class SentimentResult(TypedDict):
    sentiment: SentimentType
    confidence: float

class TranslationResult(TypedDict):
    translation: str
    metadata: Dict[str, Any]

class SummaryResult(TypedDict):
    summary: str
    metadata: ApiMetadata

# Image Types
class ImageGenerationMode(str, Enum):
    FAST = 'fast'
    STANDARD = 'standard'

class ImageGenerationOptions(TypedDict):
    mode: NotRequired[ImageGenerationMode]  # default "standard"
    negative_prompt: NotRequired[str]
    width: NotRequired[int]  # default 1024
    height: NotRequired[int]  # default 1024
    guidance: NotRequired[float]  # default 7.5
    seed: NotRequired[int]
