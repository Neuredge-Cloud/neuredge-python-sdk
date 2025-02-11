from colorama import Fore, Style
from typing import Any, Optional, Dict, List, Union
from contextlib import contextmanager
import time

def log_test_step(message: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{message}")

def log_response(name: str, data: Any, indent: int = 4) -> None:
    if isinstance(data, (dict, list)):
        from pprint import pformat
        formatted = pformat(data, indent=2)
        print(f"{' ' * indent}Response ({name}):\n{' ' * (indent+2)}{formatted}")
    else:
        print(f"{' ' * indent}Response ({name}): {data}")

def assert_with_log(condition: bool, message: str, indent: int = 4) -> None:
    if not condition:
        print(f"{' ' * indent}{Fore.RED}Assertion failed: {message}{Style.RESET_ALL}")
    assert condition, message

@contextmanager
def timing(operation: str, indent: int = 4):
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"{' ' * indent}Operation '{operation}' took {duration:.2f}s")
