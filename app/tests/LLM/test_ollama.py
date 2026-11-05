
from app.generation.providers.ollama import OllamaProvider


def main():
    provider = OllamaProvider()

    try:
        # Check Ollama connection
        if not provider.check_connection():
            print("Ollama server is not available.")
            return

        print("Ollama connected successfully.")

        # Display available models
        print("\nAvailable models:")
        for model in provider.list_models():
            print("-", model)

        # Test generation
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain what a RAG system is."
            }
        ]

        print("\nGenerating response...")

        answer = provider.generate(
            messages=messages,
            model="llama3.2:3b",
            temperature=0.2,
            max_tokens=512,
        )

        print("\nGenerated answer:")
        print(answer)

    except Exception as exc:
        print(f"Error: {exc}")

    finally:
        provider.close()


if __name__ == "__main__":
    main()


# run with
# python -m app.tests.LLM.test_ollama

