from neuredge_sdk import Neuredge
import json
from pathlib import Path
import time
import base64

def test_openai_completions(client):
    print("\n=== Testing OpenAI Chat Completions ===")
    
    # Standard completion
    completion = client.openai.chat.create(  # Fixed to match OpenAI structure
        model="@cf/meta/llama-2-7b-chat-fp16",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    print("Standard completion:", json.dumps(completion, indent=2))

    # Streaming completion
    print("\nTesting streaming completion...")
    stream = client.openai.chat.create(  # Fixed here too
        model="@cf/meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )
    
    print("Streaming response: ", end="", flush=True)
    try:
        for chunk in stream:
            if chunk and 'choices' in chunk and chunk['choices']:
                content = chunk['choices'][0].get('delta', {}).get('content', '')
                if content:
                    print(content, end="", flush=True)
    except Exception as e:
        print("\nStreaming error:", e)
    print("\n")

def test_embeddings(client):
    print("\n=== Testing Embeddings ===")
    embeddings = client.openai.embeddings.create(
        model="@cf/baai/bge-small-en-v1.5",
        input=["Hello, world!", "Testing embeddings"]
    )
    print("Embeddings:", json.dumps(embeddings['data'][0]['embedding'][:5], indent=2), "...")

def test_text_capabilities(client):
    print("\n=== Testing Text Capabilities ===")
    
    text = """Workers AI allows you to run machine learning models on the Cloudflare network.
    With the launch of Workers AI, Cloudflare is rolling out GPUs globally."""
    
    summary = client.text.summarize(text)
    print("Summary:", summary)

    translated = client.text.translate(
        text="Hello world",
        target_lang="es",
        source_lang="en"
    )
    print("Translation:", translated)

    sentiment = client.text.analyze_sentiment(
        "I love this product, it works great!"
    )
    print("Sentiment:", json.dumps(sentiment, indent=2))

def test_vector_store(client):
    print("\n=== Testing Vector Store Operations ===")
    
    index_name = f"test-vectors-{int(time.time())}"
    dimension = 384  # Using BGE small dimension

    try:
        # List existing indexes
        indexes = client.vector.list_indexes()
        print("Current indexes:", [idx["name"] for idx in indexes])

        # Create index
        print(f"Creating index: {index_name}")
        client.vector.create_index({
            "name": index_name,
            "dimension": dimension,
            "metric": "cosine"
        })
        print("Index created")

        # Wait for index to be ready
        time.sleep(2)

        # Verify index exists
        index = client.vector.get_index(index_name)
        if not index:
            raise Exception("Index was not created properly")
        print("Index verified:", index)

        # Add vectors
        vectors = [
            {
                "id": "1",
                "values": [0.1] * dimension
            },
            {
                "id": "2",
                "values": [0.2] * dimension
            }
        ]

        print("Adding vectors...")
        add_result = client.vector.add_vectors(
            index_name,
            vectors=vectors,
            options={
                "consistency": {
                    "enabled": True,
                    "maxRetries": 3,
                    "retryDelay": 1000
                }
            }
        )
        print("Vectors added:", add_result)

        # Wait for vectors to be indexed
        time.sleep(2)

        # Search vectors
        print("Searching vectors...")
        search_results = client.vector.search_vector(
            index_name,
            vector=[0.1] * dimension,
            options={
                "topK": 2,
                "consistency": {"enabled": True}
            }
        )
        print("Search results:", search_results)

        # Cleanup
        print("Cleaning up...")
        client.vector.delete_index(index_name)
        print("Index deleted successfully")

    except Exception as e:
        print("Vector store error:", e)

def test_image_generation(client):
    print("\n=== Testing Image Generation ===")
    
    try:
        # Fast generation
        print("Testing fast image generation...")
        fast_image = client.image.generate_fast(
            "A simple sketch of a cat"
        )
        print("Fast image generated:", isinstance(fast_image, bytes))
        
        # Standard generation
        print("\nTesting standard image generation...")
        standard_image = client.image.generate(
            "A magical forest with glowing mushrooms",
            options={
                "mode": "standard",
                "width": 1024,
                "height": 768,
                "guidance": 8.5,
                "negativePrompt": "dark, scary, spooky"
            }
        )
        print("Standard image generated:", isinstance(standard_image, bytes))

        # Save images
        images_dir = Path("generated_images")
        images_dir.mkdir(exist_ok=True)
        
        # Save the images directly as they're already in bytes format
        fast_image_path = images_dir / f"fast-image-{int(time.time())}.png"
        standard_image_path = images_dir / f"standard-image-{int(time.time())}.png"

        with open(fast_image_path, 'wb') as f:
            f.write(fast_image)
        print(f"Fast image saved to: {fast_image_path}")
        
        with open(standard_image_path, 'wb') as f:
            f.write(standard_image)
        print(f"Standard image saved to: {standard_image_path}")

    except Exception as e:
        print("Image generation error:", str(e))
        print("Error type:", type(e).__name__)

def main():
    client = Neuredge(
        api_key="7b5e9371-a064-4070-b5b7-659d8d80591e",
        max_retries=3,
        retry_delay=1.0
    )

    try:
        test_openai_completions(client)
        # test_embeddings(client)
        # test_text_capabilities(client)
        # test_vector_store(client)
        # test_image_generation(client)
    except Exception as e:
        print("Unexpected error:", str(e))
        print("Error type:", type(e).__name__)

if __name__ == "__main__":
    main()
