import faiss
import numpy as np

index = None
chunks = []


def create_vector_store(embeddings, text_chunks):
    global index
    global chunks

    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    chunks = text_chunks


def search(query_vector, k=3):

    distances, ids = index.search(
        np.array([query_vector]).astype("float32"),
        k
    )

    return [
        chunks[i]
        for i in ids[0]
    ]