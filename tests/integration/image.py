from dataclasses import dataclass
from neuredge_sdk import Neuredge
from tests.config import TEST_CONFIG

@dataclass
class TestCase:
    name: str
    func: callable
    description: str = ""

def test_image_generation():
    """Test image generation"""
    print("    Testing image generation...")
    client = Neuredge(**TEST_CONFIG)
    
    response = client.image.generate(
        prompt="A cute robot learning to code",
        options={
            "width": 512,
            "height": 512
        }
    )
    
    assert 'images' in response, "Response should contain images"
    assert len(response['images']) > 0, "Should have at least one image"
    assert any(img.startswith('data:image/') for img in response['images']), "Should be base64 images"
    print(f"    Generated {len(response['images'])} image(s)")

def test_image_options():
    """Test image generation with different options"""
    client = Neuredge(**TEST_CONFIG)
    
    response = client.image.generate(
        prompt="A serene landscape",
        options={
            "width": 256,
            "height": 256,
            "style": "realistic",
            "model": "@cf/stabilityai/stable-diffusion-xl-base-1.0"
        }
    )
    
    assert len(response['images']) > 0

TEST_CASES = [
    TestCase(
        name="image_generation",
        func=test_image_generation,
        description="Test basic image generation"
    ),
    TestCase(
        name="image_options",
        func=test_image_options,
        description="Test image generation options"
    )
]

def image_tests():
    """Run all image capability tests"""
    for test in TEST_CASES:
        print(f"\n  Running {test.name}...")
        try:
            test.func()
            print(f"  ✓ {test.name} passed")
        except Exception as e:
            print(f"  ✗ {test.name} failed: {str(e)}")
            raise
