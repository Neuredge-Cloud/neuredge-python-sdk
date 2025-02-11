import time
from neuredge_sdk import Neuredge, NeuredgeError
from neuredge_sdk.openai.completions import MODEL_MAPPINGS as CHAT_MODELS
from neuredge_sdk.openai.embeddings import MODEL_MAPPINGS as EMBEDDING_MODELS

def main():
    # Initialize the client with retries
    client = Neuredge(
        api_key='nrkey_239f055f1a3acc1b576e91e6489b7a6c',  # Your API key
        base_url='http://127.0.0.1:8787',  # Local development server
        max_retries=3,
        retry_delay=1.0
    )

    # Using context manager for proper cleanup
    with client:
        # Show available model mappings
        print("\nAvailable Chat Model mappings:")
        for openai_model, our_model in CHAT_MODELS.items():
            print(f"{openai_model} -> {our_model}")

        print("\nAvailable Embedding Model mappings:")
        for openai_model, our_model in EMBEDDING_MODELS.items():
            print(f"{openai_model} -> @cf/baai/bge-base-en-v1.5")

        print("\n=== Testing Text Capabilities ===")
        try:
            summary = client.text.summarize(
                "Workers AI allows you to run machine learning models, on the Cloudflare "
                "network, from your own code - whether that be from Workers, Pages, or "
                "anywhere via the Cloudflare API. With the launch of Workers AI, Cloudflare "
                "is slowly rolling out GPUs to its global network."
            )
            print("Summary:", summary)
        except NeuredgeError as e:
            if e.code == 'QUOTA_EXCEEDED':
                print("Text Summarization Error: You have exceeded your quota limit")
            else:
                print(f"Text Summarization Error: {e.code} - {str(e)}")

        try:
            sentiment = client.text.analyze_sentiment(
                "I love using the Neuredge AI platform!"
            )
            print("\nSentiment:", sentiment)
        except NeuredgeError as e:
            if e.code == 'QUOTA_EXCEEDED':
                print("Sentiment Analysis Error: You have exceeded your quota limit")
            else:
                print(f"Sentiment Analysis Error: {e.code} - {str(e)}")

        print("\n=== Testing OpenAI Compatibility ===")
        try:
            # Show which model will be used
            model = "gpt-3.5-turbo"
            mapped_model = CHAT_MODELS.get(model, model)
            print(f"Using model: {model} -> {mapped_model}")
            
            completion = client.openai.chat.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! How are you?"}
                ],
                model=model
            )
            if completion and 'choices' in completion:
                print("Chat response:", completion["choices"][0]["message"]["content"])
            else:
                print("Warning: Unexpected completion format:", completion)
        except NeuredgeError as e:
            if e.code == 'QUOTA_EXCEEDED':
                print("Chat Completion Error: You have exceeded your quota limit")
            else:
                print(f"Chat Completion Error: {e.code} - {str(e)}")

        print("\n=== Testing Vector Store Capabilities ===")
        index_name = "test-index-" + str(int(time.time()))

        try:
            # Clean up any existing index
            try:
                client.vector.delete_index(index_name)
                print("Cleaned up existing index")
            except NeuredgeError as e:
                if "not found" not in str(e).lower():
                    print(f"Warning: {str(e)}")

            # Create new index with retry
            for attempt in range(3):
                try:
                    index_config = {
                        "name": index_name,
                        "dimension": 768,  # Changed from 1536 to match BGE base model
                        "metric": "cosine"
                    }
                    client.vector.create_index(index_config)
                    print("Created vector index")
                    break
                except NeuredgeError as e:
                    if "already exists" in str(e).lower() and attempt < 2:
                        print("Retrying index creation...")
                        time.sleep(2 ** attempt)
                    else:
                        raise

            # Create embedding
            try:
                # Show which embedding model will be used
                model = "text-embedding-ada-002"
                mapped_model = EMBEDDING_MODELS.get(model, model)
                print(f"Using embedding model: {model} -> @cf/baai/bge-base-en-v1.5")
                
                embedding = client.openai.embeddings.create(
                    input="Hello world",
                    model=model
                )
                vector = embedding["data"][0]["embedding"]
                print("Created embedding")
            except NeuredgeError as e:
                print(f"Embedding Creation Error: {e.code} - {str(e)}")
                print("Using fallback dummy vector")
                vector = [0.1] * 768  # Changed from 1536 to match BGE base model

            # Add vectors with consistency mode
            vectors = [{"id": "1", "values": vector}]
            result = client.vector.add_vectors(
                index_name, 
                vectors,
                {"consistency": {"enabled": True, "max_retries": 3}}
            )
            print("Added vectors:", result)

            # Search vectors
            results = client.vector.search_vector(
                index_name,
                vector,
                {"top_k": 1}
            )
            print("Search results:", results)

        except NeuredgeError as e:
            if e.code == 'QUOTA_EXCEEDED':
                print("Vector Store Error: You have exceeded your quota limit")
            else:
                print(f"Vector Store Error: {e.code} - {str(e)}")

        finally:
            # Clean up test index
            try:
                client.vector.delete_index(index_name)
                print("Cleaned up vector index")
            except NeuredgeError as e:
                print(f"Warning during cleanup: {str(e)}")

if __name__ == "__main__":
    main()
