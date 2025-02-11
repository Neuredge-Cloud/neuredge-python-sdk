# Neuredge Python SDK

The official Python client for the Neuredge AI Platform. Provides both direct platform capabilities and OpenAI-compatible interfaces.

## Installation

```bash
pip install neuredge-sdk  # For production use
pip install -e ".[test]"  # For development with test dependencies
```

## Features

- ✨ OpenAI-compatible interfaces for chat completions and embeddings
- 📝 Text summarization, sentiment analysis, and translation
- 🖼️ Image generation with style and size options
- 🔍 Vector storage and search with consistency controls
- ⚡ Synchronous API calls for ease of use
- 🔒 Automatic retries and standardized error handling

## Quick Start

### Initialize the Client

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")
```

### Text Processing

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Summarization
summary = client.text.summarize("Long text to summarize...")
print(summary)

# Sentiment Analysis
sentiment = client.text.analyze_sentiment("I love this product!")
print(sentiment['sentiment'])

# Translation
spanish = client.text.translate(
    text="Hello, world!",
    source_lang="en",
    target_lang="es"
)
print(spanish)
```

### Chat Completions (OpenAI Compatible)

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Basic completion
completion = client.openai.chat.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    model="gpt-3.5-turbo"  # Maps to our models automatically
)
print(completion['choices'][0]['message']['content'])

# Streaming completion
for chunk in client.openai.chat.create(
    messages=[{"role": "user", "content": "Count to 3"}],
    model="gpt-3.5-turbo",
    stream=True
):
    if chunk['choices'][0]['delta'].get('content'):
        print(chunk['choices'][0]['delta']['content'], end='')
```

### Embeddings (OpenAI Compatible)

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

embedding = client.openai.embeddings.create(
    input="Hello world",
    model="text-embedding-ada-002"  # Maps to our models
)
print(embedding['data'][0]['embedding'][:5])  # First 5 dimensions
```

### Vector Store Operations

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Create index
client.vector.create_index({
    "name": "my-vectors",
    "dimension": 768,
    "metric": "cosine"
})

# Store vectors with consistency
client.vector.add_vectors(
    "my-vectors",
    vectors=[{"id": "1", "values": embedding['data'][0]['embedding']}],
    options={"consistency": {"enabled": True}}
)

# Search similar vectors
results = client.vector.search_vector(
    "my-vectors",
    vector=embedding['data'][0]['embedding'],
    options={"top_k": 10}
)
print(results[0])
```

### Image Generation

```python
from neuredge_sdk import Neuredge

client = Neuredge(api_key="your_api_key")

# Generate image
response = client.image.generate(
    prompt="A cute robot learning to code",
    options={
        "width": 512,
        "height": 512,
        "style": "realistic"
    }
)

# Base64 encoded images
for image in response['images']:
    print(f"Generated image: {image[:50]}...")
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
````
