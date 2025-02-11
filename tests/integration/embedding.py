from dataclasses import dataclass
from neuredge_sdk import Neuredge
from tests.config import TEST_CONFIG
from tests.utils import log_test_step, log_response, assert_with_log, timing

@dataclass
class TestCase:
    name: str
    func: callable
    description: str = ""

def test_single_embedding():
    with timing("single_embedding"):
        log_test_step("Testing single embedding creation...")
        client = Neuredge(**TEST_CONFIG)
        
        response = client.openai.embeddings.create(
            input="Hello world",
            model="text-embedding-ada-002"
        )
        
        log_response("Embedding", response)
        assert_with_log('data' in response, "Response should have data array")
        assert_with_log(len(response['data']) == 1, "Should have one embedding")
        assert_with_log(len(response['data'][0]['embedding']) == 768, "Should be 768-dimensional")

def test_batch_embedding():
    with timing("batch_embedding"):
        log_test_step("Testing batch embedding creation...")
        client = Neuredge(**TEST_CONFIG)
        
        response = client.openai.embeddings.create(
            input=["First text", "Second text"],
            model="text-embedding-ada-002"
        )
        
        log_response("Embedding", response)
        assert_with_log(len(response['data']) == 2, "Should have two embeddings")
        assert_with_log('usage' in response, "Should include usage stats")

TEST_CASES = [
    TestCase(
        name="single_embedding",
        func=test_single_embedding,
        description="Test single text embedding"
    ),
    TestCase(
        name="batch_embedding",
        func=test_batch_embedding,
        description="Test batch embedding"
    )
]

def embedding_tests():
    """Run all embedding capability tests"""
    for test in TEST_CASES:
        print(f"\n  Running {test.name}...")
        try:
            test.func()
            print(f"  ✓ {test.name} passed")
        except Exception as e:
            print(f"  ✗ {test.name} failed: {str(e)}")
            raise
