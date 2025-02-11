from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from ..types import ApiResponse, NeuredgeError

T = TypeVar('T')

class BaseCapability(ABC, Generic[T]):
    """Base class for all capabilities"""
    def __init__(self, client):
        self._client = client

    @property
    @abstractmethod
    def base_path(self) -> str:
        """Get the base path for this capability's endpoints"""
        pass

    def endpoint(self, path: str) -> str:
        """
        Build a full endpoint path by combining base_path and the given path
        
        Args:
            path: The endpoint-specific path
            
        Returns:
            The full endpoint path
        """
        return f"{self.base_path}{path}"

    def _handle_error(self, error: Exception) -> NeuredgeError:
        """Convert any error to NeuredgeError format"""
        if isinstance(error, NeuredgeError):
            return error
        return NeuredgeError(str(error))
