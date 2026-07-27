import faiss
import numpy as np
import pickle

index = None
chunks = []


def create_vector_store(embeddings, text_chunks):
    global index
    global chunks

    chunks = text_chunks

    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)

    vectors = np.array(embeddings).astype("float32")

    index.add(vectors)


def search(query_vector, k=3):
    global index
    global chunks

    distances, ids = index.search(
        np.array([query_vector]).astype("float32"),
        k
    )

    return [
        chunks[i]
        for i in ids[0]
    ]


def save_vector_store():
    global index
    global chunks

    faiss.write_index(
        index,
        "data/index.faiss"
    )

    with open("data/chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)


def load_vector_store():
    global index
    global chunks

    index = faiss.read_index(
        "data/index.faiss"
    )

    with open("data/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)