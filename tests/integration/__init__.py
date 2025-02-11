import sys
from dataclasses import dataclass
import colorama
from colorama import Fore, Style
from tests.config import TEST_CONFIG
from tests.utils import log_test_step
from .text import text_tests
from .completion import completion_tests
from .embedding import embedding_tests
from .vector import vector_tests
from .image import image_tests

# Initialize colorama for Windows support
colorama.init()

@dataclass
class TestResult:
    name: str
    success: bool
    error: Exception = None

def run_suite(name: str, test_fn) -> TestResult:
    print(f"\n{Fore.BLUE}{Style.BRIGHT}=== Running {name} Test Suite ==={Style.RESET_ALL}")
    try:
        # Remove async/await completely
        test_fn()
        print(f"{Fore.GREEN}✓ {name} suite completed{Style.RESET_ALL}")
        return TestResult(name=name, success=True)
    except Exception as e:
        print(f"{Fore.RED}✗ {name} suite failed: {str(e)}{Style.RESET_ALL}")
        return TestResult(name=name, success=False, error=e)

def run_all_tests() -> bool:
    print(f"{Style.BRIGHT}\nStarting all integration tests...\n{Style.RESET_ALL}")
    
    if not TEST_CONFIG['api_key'] or TEST_CONFIG['api_key'] == 'test_key':
        print(f"{Fore.RED}Please configure a valid API key in tests/config.py{Style.RESET_ALL}")
        sys.exit(1)

    results = [
        run_suite('Text', text_tests),
        run_suite('Completion', completion_tests),
        run_suite('Embedding', embedding_tests),
        run_suite('Vector', vector_tests),
        run_suite('Image', image_tests)
    ]

    # Print summary
    print(f"\n{Style.BRIGHT}Test Summary:{Style.RESET_ALL}")
    failed_tests = [r for r in results if not r.success]
    
    for result in results:
        status = f"{Fore.GREEN}PASS" if result.success else f"{Fore.RED}FAIL"
        print(f"{status}{Style.RESET_ALL} - {result.name}")
        if not result.success and result.error:
            print(f"  └─ {Fore.RED}{str(result.error)}{Style.RESET_ALL}")
    
    print(f"\nResults: {len(results) - len(failed_tests)}/{len(results)} tests passed")
    sys.exit(0 if len(failed_tests) == 0 else 1)  # Exit with proper status code

if __name__ == "__main__":
    print(f"{Style.BRIGHT}Integration Tests{Style.RESET_ALL}")
    try:
        success = run_all_tests()
    except Exception as e:
        print(f"{Fore.RED}\nTest runner failed: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
