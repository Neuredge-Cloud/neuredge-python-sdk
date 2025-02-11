import asyncio
import sys
from colorama import init, Fore, Style
from tests.config import TEST_CONFIG
from tests.integration import run_all_tests

# Initialize colorama
init()

if __name__ == "__main__":
    try:
        # Validate API key first
        if not TEST_CONFIG['api_key'] or TEST_CONFIG['api_key'] == 'test_key':
            print(f"{Fore.RED}Please configure a valid API key in tests/config.py{Style.RESET_ALL}")
            sys.exit(1)

        asyncio.run(run_all_tests())
    except Exception as e:
        print(f"\n{Fore.RED}Test runner failed: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
