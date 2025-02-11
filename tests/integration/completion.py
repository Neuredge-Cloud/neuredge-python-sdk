from dataclasses import dataclass
from neuredge_sdk import Neuredge
from tests.config import TEST_CONFIG
from tests.utils import log_test_step, log_response, assert_with_log, timing

@dataclass
class TestCase:
    name: str
    func: callable
    description: str = ""

def test_basic_completion():
    with timing("basic_completion"):
        log_test_step("Testing basic chat completion...")
        client = Neuredge(**TEST_CONFIG)
        
        response = client.openai.chat.create(
            messages=[{"role": "user", "content": "Say hello!"}],
            model="gpt-3.5-turbo"
        )
        
        log_response("Chat", response)
        assert_with_log('choices' in response, "Response should have choices")
        assert_with_log(len(response['choices']) > 0, "Should have at least one choice")
        assert_with_log('content' in response['choices'][0]['message'], "Should have message content")

def test_system_message():
    with timing("system_message"):
        log_test_step("Testing system message handling...")
        client = Neuredge(**TEST_CONFIG)
        
        response = client.openai.chat.create(
            messages=[
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello!"}
            ],
            model="gpt-3.5-turbo"
        )
        
        log_response("Chat", response)
        assert_with_log('choices' in response, "Response should have choices")

def test_stream_completion():
    """Test streaming chat completion"""
    with timing("stream_completion"):
        log_test_step("Testing streaming chat completion...")
        client = Neuredge(**TEST_CONFIG)
        
        stream = client.openai.chat.create(
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
            model="gpt-3.5-turbo",
            stream=True
        )
        
        log_test_step("Receiving stream chunks:")
        full_response = []
        for chunk in stream:
            if 'choices' in chunk and chunk['choices'][0].get('delta', {}).get('content'):
                content = chunk['choices'][0]['delta']['content']
                log_test_step(f"Chunk: {content}")
                full_response.append(content)
        
        complete_text = ''.join(full_response)
        log_test_step(f"Complete response: {complete_text}")
        assert_with_log(len(full_response) > 0, "Should receive multiple chunks")

TEST_CASES = [
    TestCase(
        name="basic_completion",
        func=test_basic_completion,
        description="Test basic chat completion"
    ),
    TestCase(
        name="system_message",
        func=test_system_message,
        description="Test system message handling"
    ),
    TestCase(
        name="stream_completion",
        func=test_stream_completion,
        description="Test streaming chat completion"
    )
]

def completion_tests():
    """Run all completion capability tests"""
    for test in TEST_CASES:
        print(f"\n  Running {test.name}...")
        try:
            test.func()
            print(f"  ✓ {test.name} passed")
        except Exception as e:
            print(f"  ✗ {test.name} failed: {str(e)}")
            raise
