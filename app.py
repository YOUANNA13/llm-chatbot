import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import get_embedding, get_embeddings_batch
from vector_store import VectorStore
import os
import base64
import uuid

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="assets/logo.png",
    layout="wide"
)

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

# ---------------------------------------------------------
# Theme colors (phosphoric green / black)
# ---------------------------------------------------------
PHOSPHOR_GREEN = "#39FF14"
DARK_BG = "#050505"
PANEL_BG = "#0d0d0d"


def get_base64_image(path):
    """Read a local image and return its base64 string, or None if missing."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


logo_base64 = get_base64_image("assets/logo.png")

# ---------------------------------------------------------
# Session identity + persistence.
#
# IMPORTANT: this must run BEFORE the splash screen below. Writing to
# st.query_params (to store a persistent ?sid=... token) can trigger an
# extra script rerun. If that rerun happened AFTER the splash markdown
# was inserted, Streamlit reconciles it away almost instantly - the
# splash element gets added then removed within milliseconds, so it
# never becomes visible. Running this block first avoids that.
#
# A hard browser refresh normally wipes st.session_state entirely, which
# would mean re-uploading PDFs every time. To survive a refresh, we give
# each browser tab a persistent token in the URL (?sid=...) and use it as
# the name of a per-session folder on disk.
# ---------------------------------------------------------
SESSIONS_DIR = "data/sessions"

if "session_id" not in st.session_state:
    existing_sid = st.query_params.get("sid")
    if existing_sid:
        st.session_state.session_id = existing_sid
    else:
        new_sid = uuid.uuid4().hex[:12]
        st.session_state.session_id = new_sid
        st.query_params["sid"] = new_sid

session_id = st.session_state.session_id
session_dir = os.path.join(SESSIONS_DIR, session_id)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    st.session_state.vector_store.load(session_dir)

if "messages" not in st.session_state:
    st.session_state.messages = []

vector_store = st.session_state.vector_store

# ---------------------------------------------------------
# Global theme CSS
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK_BG};
        color: #e8ffe8;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL_BG};
        border-right: 1px solid {PHOSPHOR_GREEN}33;
    }}
    h1, h2, h3 {{
        color: {PHOSPHOR_GREEN};
        text-shadow: 0 0 8px {PHOSPHOR_GREEN}55;
    }}
    .stCaption, p, span, label {{
        color: #d8ffd8 !important;
    }}
    .stButton > button, .stDownloadButton > button {{
        background-color: transparent;
        color: {PHOSPHOR_GREEN};
        border: 1px solid {PHOSPHOR_GREEN};
        border-radius: 8px;
        box-shadow: 0 0 6px {PHOSPHOR_GREEN}44;
        transition: all 0.2s ease-in-out;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {PHOSPHOR_GREEN}22;
        box-shadow: 0 0 14px {PHOSPHOR_GREEN}aa;
    }}
    [data-testid="stChatInput"] {{
        border: 1px solid {PHOSPHOR_GREEN}66;
        box-shadow: 0 0 10px {PHOSPHOR_GREEN}22;
        border-radius: 10px;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border: 1px solid {PHOSPHOR_GREEN} !important;
        box-shadow: 0 0 16px {PHOSPHOR_GREEN}aa !important;
        outline: none !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {PHOSPHOR_GREEN} !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stFileUploader"] {{
        border: 1px dashed {PHOSPHOR_GREEN};
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 0 8px {PHOSPHOR_GREEN}33;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: transparent;
    }}
    [data-testid="stChatMessage"] {{
        background-color: {PANEL_BG};
        border: 1px solid {PHOSPHOR_GREEN}22;
        border-radius: 10px;
    }}
    ::selection {{
        background: {PHOSPHOR_GREEN}55;
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 0.55; text-shadow: 0 0 6px {PHOSPHOR_GREEN}55; }}
        50% {{ opacity: 1; text-shadow: 0 0 16px {PHOSPHOR_GREEN}cc; }}
    }}
    #empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 90px 20px;
        opacity: 0.9;
    }}
    #empty-state-icon {{
        font-size: 42px;
        margin-bottom: 14px;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }}
    #empty-state-title {{
        color: {PHOSPHOR_GREEN};
        font-size: 18px;
        font-weight: 600;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }}
    #empty-state-sub {{
        color: #9fdca0;
        font-size: 13px;
        margin-top: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Splash screen (shows once per session, shrinks to top-left)
# ---------------------------------------------------------
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if logo_base64 and not st.session_state.splash_done:
    st.markdown(
        f"""
        <style>
        @keyframes shrinkMoveFade {{
            0% {{
                top: 50%; left: 50%;
                transform: translate(-50%, -50%) scale(1);
                opacity: 1;
            }}
            70% {{
                top: 80px; left: 60px;
                transform: translate(0, 0) scale(0.4);
                opacity: 1;
            }}
            100% {{
                top: 80px; left: 60px;
                transform: translate(0, 0) scale(0.4);
                opacity: 0;
                visibility: hidden;
            }}
        }}
        @keyframes fadeOutOverlay {{
            to {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        #splash-overlay {{
            position: fixed;
            inset: 0;
            background-color: {DARK_BG};
            z-index: 999999;
            animation: fadeOutOverlay 0.5s ease forwards;
            animation-delay: 2.4s;
        }}
        #splash-logo {{
            position: fixed;
            width: 240px;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            filter: drop-shadow(0 0 30px {PHOSPHOR_GREEN});
            animation: shrinkMoveFade 1.3s ease forwards;
            animation-delay: 1.1s;
            z-index: 1000000;
        }}
        </style>
        <div id="splash-overlay"></div>
        <img id="splash-logo" src="data:image/png;base64,{logo_base64}" />
        """,
        unsafe_allow_html=True
    )
    st.session_state.splash_done = True

# ---------------------------------------------------------
# App content
# ---------------------------------------------------------
st.title("AI Document Assistant")
st.caption("Ask questions across one or more PDFs using Gemini and RAG.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not st.session_state.messages:
    if not vector_store.has_documents():
        st.markdown(
            """
            <div id="empty-state">
                <div id="empty-state-icon">&#128196;</div>
                <div id="empty-state-title">Upload one or more PDFs to get started</div>
                <div id="empty-state-sub">Your documents will be indexed, then you can ask anything about them.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div id="empty-state">
                <div id="empty-state-icon">&#128172;</div>
                <div id="empty-state-title">Ask a question about your documents</div>
                <div id="empty-state-sub">Try something like &quot;summarize the key points&quot;.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# Sidebar: branding + multi-file upload
# ---------------------------------------------------------
if logo_base64:
    st.sidebar.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <img src="data:image/png;base64,{logo_base64}"
                 style="width:90px; filter: drop-shadow(0 0 6px {PHOSPHOR_GREEN}aa); display:block;" />
            <div style="margin-top:8px; font-size:12px; letter-spacing:2px;
                        color:{PHOSPHOR_GREEN}; text-shadow: 0 0 4px {PHOSPHOR_GREEN}88;">
                AI &bull; CODE &bull; PDF
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.title("Document")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Only index files that aren't already in the store, so re-running
    # the script (e.g. after asking a question) doesn't re-embed
    # everything from scratch every time.
    new_files = [
        f for f in uploaded_files
        if f.name not in vector_store.indexed_files
    ]

    if new_files:

        with st.spinner(f"Indexing {len(new_files)} document(s)..."):

            # Gather every new file's chunks first, so embedding can be
            # batched ONCE across all of them together, instead of running
            # a separate batching pass per file (which fires bursts of
            # requests close together and is more likely to hit the
            # per-minute rate limit when several PDFs are uploaded at once).
            all_chunks = []
            file_ranges = []  # (filename, start_index, end_index)

            for uploaded_file in new_files:

                pages = load_pdf(uploaded_file)
                chunks = chunk_text(pages, filename=uploaded_file.name)

                start = len(all_chunks)
                all_chunks.extend(chunks)
                end = len(all_chunks)

                file_ranges.append((uploaded_file.name, start, end))

            all_embeddings = get_embeddings_batch(
                client,
                [chunk["text"] for chunk in all_chunks]
            )

            for filename, start, end in file_ranges:
                vector_store.add_documents(
                    all_embeddings[start:end],
                    all_chunks[start:end],
                    filename=filename
                )

            vector_store.save(session_dir)

        st.sidebar.success(f"Indexed {len(new_files)} new document(s).")

if vector_store.indexed_files:
    with st.sidebar.expander(f"Indexed documents ({len(vector_store.indexed_files)})", expanded=False):
        for name in sorted(vector_store.indexed_files):
            st.write(f"- {name}")

    if st.sidebar.button("Clear all documents"):
        vector_store.clear(session_dir)
        st.session_state.messages = []
        st.rerun()

# Let the user scope a question to specific documents instead of always
# searching across everything that's been uploaded.
selected_files = None
if len(vector_store.indexed_files) > 1:
    selected_files = st.sidebar.multiselect(
        "Search within",
        options=sorted(vector_store.indexed_files),
        default=sorted(vector_store.indexed_files)
    )

# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------
question = st.chat_input("Ask anything about the uploaded document(s)...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    contents = []

    retrieved_chunks = []

    if vector_store.has_documents():

        question_embedding = get_embedding(client, question)

        retrieved_chunks = vector_store.search(
            question_embedding,
            k=3,
            filenames=selected_files
        )

        context = "\n\n".join(
            f"[{chunk['filename']}, page {chunk['page']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

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

        placeholder.markdown(
            f"""
            <span style="color:{PHOSPHOR_GREEN}; animation: pulseGlow 1.2s ease-in-out infinite;">
                &#9679; thinking...
            </span>
            """,
            unsafe_allow_html=True
        )

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

        if retrieved_chunks:

            # Group sources by filename -> sorted page numbers, since
            # chunks can now come from multiple documents.
            sources_by_file = {}
            for chunk in retrieved_chunks:
                sources_by_file.setdefault(chunk["filename"], set()).add(chunk["page"])

            full_response += "\n\n### Sources\n"

            for filename, pages in sources_by_file.items():
                page_list = ", ".join(str(p) for p in sorted(pages))
                full_response += f"- **{filename}** — page(s) {page_list}\n"

            full_response += f"\nRetrieved {len(retrieved_chunks)} chunks."

            placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })