from dataclasses import dataclass
import time
from neuredge_sdk import Neuredge
from tests.config import TEST_CONFIG
from tests.utils import log_test_step, log_response, assert_with_log, timing

@dataclass
class TestCase:
    name: str
    func: callable
    description: str = ""

def test_index_operations():
    with timing("index_operations"):
        log_test_step("Testing vector index operations...")
        client = Neuredge(**TEST_CONFIG)
        index_name = f"test-index-{int(time.time())}"
        
        # Create index
        client.vector.create_index({
            "name": index_name,
            "dimension": 768,
            "metric": "cosine"
        })
        log_test_step("Created index")
        
        # Verify and cleanup
        index = client.vector.get_index(index_name)
        log_response("Index", index)
        assert_with_log(index is not None, "Index should exist")

async def test_vector_operations():
    """Test vector storage and search"""
    client = Neuredge(**TEST_CONFIG)
    index_name = f"test-vectors-{int(time.time())}"
    
    try:
        # Setup
        await client.vector.create_index({
            "name": index_name,
            "dimension": 768,
            "metric": "cosine"
        })
        
        # Add vectors
        vector = [0.1] * 768
        result = await client.vector.add_vectors(
            index_name,
            [{"id": "1", "values": vector}],
            {"consistency": {"enabled": True}}
        )
        assert result['inserted'] == 1, "Should insert one vector"
        
        # Search
        matches = await client.vector.search_vector(
            index_name,
            vector,
            {"top_k": 1}
        )
        assert len(matches) == 1, "Should find one match"
        assert matches[0]['id'] == "1", "Should match inserted vector"
        
    finally:
        # Cleanup
        await client.vector.delete_index(index_name)

TEST_CASES = [
    TestCase(
        name="index_operations",
        func=test_index_operations,
        description="Test vector index CRUD operations"
    ),
    TestCase(
        name="vector_operations",
        func=test_vector_operations,
        description="Test vector storage and search"
    )
]

def vector_tests():
    client = Neuredge(**TEST_CONFIG)
    index_name = f"test-vectors-{int(time.time())}"
    
    try:
        # Create index
        client.vector.create_index({
            "name": index_name,
            "dimension": 768,
            "metric": "cosine"
        })
        
        # Add vectors
        vector = [0.1] * 768
        result = client.vector.add_vectors(
            index_name,
            [{"id": "1", "values": vector}],
            {"consistency": {"enabled": True}}
        )
        assert result['inserted'] == 1
    finally:
        client.vector.delete_index(index_name)
