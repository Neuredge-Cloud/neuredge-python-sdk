from dataclasses import dataclass
from neuredge_sdk import Neuredge
from tests.config import TEST_CONFIG
from tests.utils import log_test_step, log_response, assert_with_log, timing

@dataclass
class TestCase:
    name: str
    func: callable
    description: str = ""

def test_summarization():
    """Test text summarization capability"""
    with timing("summarization"):
        log_test_step("Testing text summarization...")
        client = Neuredge(**TEST_CONFIG)
        
        text = """
        Workers AI allows you to run machine learning models on the Cloudflare network.
        With the launch of Workers AI, Cloudflare is rolling out GPUs globally.
        This enables running AI workloads at the edge, closer to users.
        """
        log_test_step(f"Input length: {len(text)} chars")
        
        summary = client.text.summarize(text)
        log_response("Summary", summary)
        
        assert_with_log(isinstance(summary, str), "Summary should be a string")
        assert_with_log(len(summary) < len(text), "Summary should be shorter than input")

def test_sentiment_analysis():
    """Test sentiment analysis capability"""
    with timing("sentiment_analysis"):
        log_test_step("Testing sentiment analysis...")
        client = Neuredge(**TEST_CONFIG)
        
        # Test positive sentiment
        text = "I love using this AI platform!"
        log_test_step(f"Testing positive text: '{text}'")
        positive = client.text.analyze_sentiment(text)
        log_response("Positive sentiment", positive)
        assert_with_log(positive['sentiment'] == 'POSITIVE', "Should detect positive sentiment")
        
        # Test negative sentiment
        text = "This is terrible and frustrating."
        log_test_step(f"Testing negative text: '{text}'")
        negative = client.text.analyze_sentiment(text)
        log_response("Negative sentiment", negative)
        assert_with_log(negative['sentiment'] == 'NEGATIVE', "Should detect negative sentiment")

def test_translation():
    """Test translation capability"""
    with timing("translation"):
        log_test_step("Testing translation...")
        client = Neuredge(**TEST_CONFIG)
        
        text = "Hello, how are you?"
        log_test_step(f"Original (en): '{text}'")
        
        # English to Spanish
        spanish = client.text.translate(text, source_lang="en", target_lang="es")
        log_response("Spanish", spanish)
        assert_with_log(spanish and isinstance(spanish, str), "Should return Spanish translation")
        
        # English to French
        french = client.text.translate(text, source_lang="en", target_lang="fr")
        log_response("French", french)
        assert_with_log(french and isinstance(french, str), "Should return French translation")

TEST_CASES = [
    TestCase(
        name="summarization",
        func=test_summarization,
        description="Test text summarization with different lengths"
    ),
    TestCase(
        name="sentiment",
        func=test_sentiment_analysis,
        description="Test sentiment analysis with different inputs"
    ),
    TestCase(
        name="translation",
        func=test_translation,
        description="Test translation between different languages"
    )
]

def text_tests():
    """Run all text capability tests"""
    for test in TEST_CASES:
        print(f"\n  Running {test.name}...")
        try:
            test.func()
            print(f"  ✓ {test.name} passed")
        except Exception as e:
            print(f"  ✗ {test.name} failed: {str(e)}")
            raise
