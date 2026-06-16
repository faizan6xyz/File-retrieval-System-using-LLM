import os
os.environ["FASTEMBED_CACHE_PATH"] = r"C:\Users\faiza\fastembed_models"
import numpy as np
import faiss
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
def build_index(file_name,
                folder_path="../Data",
                chunk_path="../",
                index_path="rag_index.faiss",
                chunks_path="chunks.npy"):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"File loaded: {file_name}")
    full_index_path  = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, chunks_path)
    model = TextEmbedding("BAAI/bge-base-en-v1.5")  # same model, no hub needed
    new_embedding = np.array(list(model.embed([text])), dtype=np.float32)
    print("Embedding shape:", new_embedding.shape)
    faiss.normalize_L2(new_embedding)
    dimension = new_embedding.shape[1]
    if os.path.exists(full_index_path):
        index = faiss.read_index(full_index_path)
        print(f"Loaded existing FAISS index ({index.ntotal} vectors)")
    else:
        index = faiss.IndexFlatIP(dimension)
        print("Created new FAISS index")
    index.add(new_embedding)
    faiss.write_index(index, full_index_path)
    print("Vectors stored:", index.ntotal)
    existing_chunks = np.load(full_chunks_path, allow_pickle=True).tolist() \
                      if os.path.exists(full_chunks_path) else []
    existing_chunks.append(text)
    np.save(full_chunks_path, np.array(existing_chunks))
    print(f"chunks.npy updated → {len(existing_chunks)} chunk(s) total")
    bm25 = BM25Okapi([chunk.lower().split() for chunk in existing_chunks])
    print("BM25 index rebuilt over all chunks.")
    return bm25
if __name__ == "__main__":
    build_index("my_document.txt")