import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import get_embedding
from vector_store import (
    create_vector_store,
    search,
    save_vector_store,
    load_vector_store
)
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

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# Load saved vector store if it exists
if (
    os.path.exists("data/index.faiss")
    and
    os.path.exists("data/chunks.pkl")
    and
    not st.session_state.pdf_ready
):
    load_vector_store()
    st.session_state.pdf_ready = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file is not None and not st.session_state.pdf_ready:

    with st.spinner("Processing PDF..."):

        pdf_text = load_pdf(uploaded_file)

        chunks = chunk_text(pdf_text)

        embeddings = []

        for chunk in chunks:
            embeddings.append(
                get_embedding(client, chunk)
            )

        create_vector_store(
            embeddings,
            chunks
        )

        save_vector_store()

        st.session_state.pdf_ready = True

    st.success("PDF processed successfully!")

question = st.chat_input("Ask anything")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    contents = []

    if st.session_state.pdf_ready:

        question_embedding = get_embedding(
            client,
            question
        )

        retrieved_chunks = search(question_embedding)

        context = "\n\n".join(retrieved_chunks)

        contents.append({
            "role": "user",
            "parts": [{
                "text": f"""
Use ONLY the following context to answer the user's question.

Context:

{context}
"""
            }]
        })

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