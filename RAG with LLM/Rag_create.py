import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    new_embedding = model.encode([text], convert_to_numpy=True, )
    print("Embedding shape:", new_embedding.shape)
    faiss.normalize_L2(new_embedding)
    dimension = new_embedding.shape[1]
    if os.path.exists(full_index_path):
        index = faiss.read_index(full_index_path)
        print(f"Loaded existing FAISS index ({index.ntotal} vectors)")
    else:
        index = faiss.IndexFlatIP(dimension)
        print("Created new FAISS index")
    index.add(new_embedding.astype(np.float32))
    print("Vectors stored:", index.ntotal)
    faiss.write_index(index, full_index_path)
    if os.path.exists(full_chunks_path):
        existing_chunks = np.load(full_chunks_path, allow_pickle=True).tolist()
        print(f"Loaded {len(existing_chunks)} existing chunk(s)")
    else:
        existing_chunks = []
        print("No existing chunks found, starting fresh")
    existing_chunks.append(text)
    np.save(full_chunks_path, np.array(existing_chunks))
    print(f"chunks.npy updated → {len(existing_chunks)} chunk(s) total")
    tokenized_docs = [chunk.lower().split() for chunk in existing_chunks]
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25 index rebuilt over all chunks.")
if __name__ == "__main__":
    build_index("my_document.txt")