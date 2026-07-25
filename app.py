import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pdf_loader import load_pdf
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt = """
You are an AI tutor.

Rules:
- Explain concepts simply.
- Give examples whenever possible.
- Keep answers concise.
- If you don't know the answer, say so.
"""

st.title("LLM ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

pdf_text = ""

if uploaded_file is not None:
    pdf_text = load_pdf(uploaded_file)

question = st.chat_input("Ask anything")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    contents = []

    # Add the PDF as the first message
    if pdf_text:
        contents.append({
            "role": "user",
            "parts": [{
                "text": f"""
This is the content of the uploaded PDF.

{pdf_text}

Answer the user's questions using this document.
"""
            }]
        })

    # Add chat history
    for message in st.session_state.messages:
        contents.append({
            "role": message["role"],
            "parts": [{
                "text": message["content"]
            }]
        })

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        response = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )

        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })