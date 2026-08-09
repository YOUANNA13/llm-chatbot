import faiss
import numpy as np
import pickle
import os


class VectorStore:
    """
    A per-session, in-memory vector store.

    IMPORTANT: create one instance per user session (e.g. stored in
    st.session_state) rather than a shared module-level instance. The old
    version of this file used module-level globals plus a single shared
    file on disk (data/index.faiss / data/chunks.pkl) - that meant every
    user of the app shared the SAME index, so the last person to upload a
    PDF silently overwrote everyone else's documents. This class fixes
    that by keeping state as instance attributes instead.

    Supports adding multiple documents incrementally, so a session can
    hold several PDFs at once instead of being replaced on each upload.
    """

    def __init__(self):
        self.index = None
        self.dimension = None
        self.chunks = []          # list of {"text", "page", "filename"}
        self.indexed_files = set()  # filenames already embedded, to avoid re-indexing

    def has_documents(self):
        return self.index is not None and self.index.ntotal > 0

    def add_documents(self, embeddings, text_chunks, filename=None):
        """
        Add new embeddings/chunks to the store.

        Creates the FAISS index on first call, and grows it on every
        subsequent call - this is what allows multiple PDFs to accumulate
        in the same store instead of replacing one another.
        """

        if not embeddings:
            return

        vectors = np.array(embeddings).astype("float32")

        if self.index is None:
            self.dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(vectors)
        self.chunks.extend(text_chunks)

        if filename:
            self.indexed_files.add(filename)

    def search(self, query_vector, k=3, filenames=None):
        """
        Search the store for the k most relevant chunks.

        If `filenames` is provided (a set/list of filenames), only chunks
        from those files are returned. FAISS's flat index has no native
        metadata filtering, so we over-fetch candidates and filter in
        Python, then trim back down to k.
        """

        if not self.has_documents():
            return []

        fetch_k = self.index.ntotal if filenames else k
        fetch_k = max(1, min(fetch_k, self.index.ntotal))

        distances, ids = self.index.search(
            np.array([query_vector]).astype("float32"),
            fetch_k
        )

        results = [
            self.chunks[i]
            for i in ids[0]
            if i != -1
        ]

        if filenames:
            results = [
                chunk for chunk in results
                if chunk.get("filename") in filenames
            ]

        return results[:k]

    def save(self, directory):
        """
        Persist this session's index + chunks to its own directory, so it
        survives a browser refresh. Each session must use a different
        directory (see app.py) - saving is what makes refresh-persistence
        possible, but it must stay per-session or we're back to the
        original shared-file bug.
        """
        if self.index is None:
            return

        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "indexed_files": self.indexed_files,
                },
                f
            )

    def load(self, directory):
        """
        Restore a previously saved session store, if one exists on disk.
        Returns True if something was loaded, False otherwise (e.g. first
        visit, or a brand new session with nothing saved yet).
        """
        index_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.pkl")

        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return False

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.indexed_files = data["indexed_files"]
        self.dimension = self.index.d

        return True

    def clear(self, directory=None):
        """Reset in-memory state, and optionally delete the saved files too."""
        self.index = None
        self.dimension = None
        self.chunks = []
        self.indexed_files = set()

        if directory:
            for name in ("index.faiss", "chunks.pkl"):
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    os.remove(path)