from typing import Callable, Any, TypeVar

# Functions
F = TypeVar('F', bound=Callable[..., Any])
