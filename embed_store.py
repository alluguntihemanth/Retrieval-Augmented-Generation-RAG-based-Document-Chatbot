import faiss
import numpy as np
import os
import pickle

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

def store_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks

def load_embeddings():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None

def retrieve_chunks(query, model, index, chunk_texts, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n\n".join([chunk_texts[i] for i in I[0]])
