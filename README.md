# Neuredge Python SDK

The official Python client for the Neuredge AI Platform.

## Installation

```bash
pip install neuredge-sdk
```

## Features

- 🤖 OpenAI-compatible chat completions and embeddings
- 📝 Text summarization, sentiment analysis, and translation
- 🎨 Image generation with fast and standard modes
- 🔍 Vector storage with consistency and retry controls
- 🔒 Built-in error handling and retries

## Quick Start

```python
from neuredge_sdk import Neuredge

client = Neuredge(
    api_key="your_api_key",
    base_url="https://api.neuredge.dev",  # Optional
    max_retries=3,                        # Optional
    retry_delay=1.0                       # Optional
)
```

### Text Processing

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Summarization
text = """Workers AI allows you to run machine learning models on the Cloudflare network.
With the launch of Workers AI, Cloudflare is rolling out GPUs globally."""
summary = client.text.summarize(text)
print(summary)

# Sentiment Analysis with actual response format
sentiment = client.text.analyze_sentiment("I love this product!")
print(f"Sentiment: {sentiment['sentiment']}")  # POSITIVE or NEGATIVE
print(f"Confidence: {sentiment['confidence']}")
print(f"Is Confident: {sentiment['is_confident']}")

# Translation
spanish = client.text.translate(
    text="Hello, world!",
    target_lang="es",
    source_lang="en"  # Optional
)
print(spanish)
```

### Chat Completions (OpenAI Compatible)

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Basic completion with actual model names
completion = client.openai.chat.create(
    model="@cf/meta/llama-2-7b-chat-fp16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(completion['choices'][0]['message']['content'])

# Streaming with actual response format
stream = client.openai.chat.create(
    model="@cf/meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Count to 3"}],
    stream=True
)
for chunk in stream:
    if chunk['choices'][0]['delta'].get('content'):
        print(chunk['choices'][0]['delta']['content'], end='')
```

### Embeddings (OpenAI Compatible)

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Using correct model name and dimensions
embedding = client.openai.embeddings.create(
    input="Hello world",
    model="@cf/baai/bge-small-en-v1.5"  # 384 dimensions
)
vector = embedding['data'][0]['embedding']  # 384-dimensional vector
print(vector[:5])  # First 5 dimensions
```

### Vector Store Operations

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Create index with correct dimensions
client.vector.create_index({
    "name": "my-vectors",
    "dimension": 384,  # BGE small dimension
    "metric": "cosine"
})

# Store vectors with proper consistency options
result = client.vector.add_vectors(
    "my-vectors",
    vectors=[{
        "id": "1",
        "values": [0.1] * 384  # Match BGE small dimensions
    }],
    options={
        "consistency": {
            "enabled": True,
            "maxRetries": 3,
            "retryDelay": 1000
        }
    }
)

# Search with actual response format
matches = client.vector.search_vector(
    "my-vectors",
    vector=[0.1] * 384,
    options={
        "topK": 10,
        "consistency": {"enabled": True}
    }
)
for match in matches:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

### Image Generation

```python
from neuredge_sdk import Neuredge
from pathlib import Path

client = Neuredge(api_key="your_api_key")

# Fast mode - Quick generation
fast_image = client.image.generate_fast(
    "A simple sketch of a cat"
)  # Returns bytes

# Standard mode with options
standard_image = client.image.generate(
    "A magical forest with glowing mushrooms",
    options={
        "mode": "standard",
        "width": 1024,
        "height": 768,
        "guidance": 8.5,
        "negativePrompt": "dark, scary, spooky"
    }
)  # Returns bytes

# Save the generated images
images_dir = Path("generated_images")
images_dir.mkdir(exist_ok=True)

# Direct file writing (bytes response)
with open(images_dir / "fast.png", 'wb') as f:
    f.write(fast_image)

with open(images_dir / "standard.png", 'wb') as f:
    f.write(standard_image)
```

### Return Types

```python
# All image generation methods return bytes
image: bytes = client.image.generate_fast("prompt")  # Direct bytes response
image: bytes = client.image.generate("prompt", options)  # Direct bytes response

# Image generation options
options = {
    "mode": "standard",    # 'fast' or 'standard'
    "width": 1024,        # 512-1024px
    "height": 1024,       # 512-1024px
    "guidance": 7.5,      # 1-20, controls prompt adherence
    "negativePrompt": ""  # Things to avoid in generation
}
```

### Response Format

```python
# Image generation response format
{
    'images': [
        'data:image/jpeg;base64,<base64-encoded-image-data>',
        # More images if batch generation
    ]
}
```

## Error Handling

```python
from neuredge_sdk import Neuredge, NeuredgeError

try:
    client = Neuredge(api_key="invalid-key")
    summary = client.text.summarize("Some text")
except NeuredgeError as e:
    if e.code == 'AUTHENTICATION_ERROR':
        print("Invalid API key")
    elif e.code == 'QUOTA_EXCEEDED':
        print("Rate limit reached")
    else:
        print(f"Error: {e.code} - {e.message}")
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
python -m tests.run_tests

# Run specific suite
python -m tests.integration.text
python -m tests.integration.vector
```

### Project Structure

```
neuredge_sdk/
├── capabilities/     # Core API capabilities
│   ├── text.py
│   ├── image.py
│   └── vector.py
├── openai/          # OpenAI-compatible interfaces
│   ├── completions.py
│   └── embeddings.py
├── types.py         # Type definitions
└── client.py        # Main client

tests/
├── integration/     # Integration tests
├── utils.py        # Test utilities
└── config.py       # Test configuration
```

## Contributing

Contributions are welcome! Please see our [Contribution Guidelines](CONTRIBUTING.md) for more information.

## License

MIT
