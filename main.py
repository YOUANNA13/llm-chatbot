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

print("=" * 50)
print("🤖 Simple Gemini ChatBot")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=question
        )

        print("\nAssistant:")
        print(response.text)

    except Exception as e:
        print(f"\nError: {e}")