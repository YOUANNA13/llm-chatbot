from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file!")
    exit()

# Create Gemini client
client = genai.Client(api_key=api_key)

# System Prompt
system_prompt = """
You are an AI tutor.

Rules:
- Explain concepts simply.
- Give examples whenever possible.
- Keep answers concise.
- If you don't know the answer, say so.
"""

# Store conversation history
history = []

print("=" * 50)
print("Simple Gemini ChatBot")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    # Get user input
    question = input("\nYou: ")

    # Exit program
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Save user message
    history.append({
        "role": "user",
        "text": question
    })

    # Convert history to Gemini format
    contents = []

    for message in history:
        contents.append({
            "role": message["role"],
            "parts": [
                {
                    "text": message["text"]
                }
            ]
        })

    try:
        print("\nAssistant: ", end="", flush=True)

        response = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )

        full_response = ""

        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text

        print()

        # Save assistant response
        history.append({
            "role": "assistant",
            "text": full_response
        })

    except Exception as e:
        print(f"\nError: {e}")