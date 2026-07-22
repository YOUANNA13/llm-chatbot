from google import genai
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

# Store conversation history
history = []

print("=" * 50)
print("🤖 Simple Gemini ChatBot")
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

    # Convert our history to Gemini format
    contents = []

    for message in history:
        contents.append(
            {
                "role": message["role"],
                "parts": [
                    {
                        "text": message["text"]
                    }
                ]
            }
        )

    try:
        # Send the ENTIRE conversation
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents
        )

        print("\nAssistant:")
        print(response.text)

        # Save assistant response
        history.append({
            "role": "assistant",
            "text": response.text
        })

    except Exception as e:
        print(f"\nError: {e}")