import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
def build_index(file_name, 
                folder_path="../Data", 
                index_path="rag_index.faiss", 
                chunks_path="chunks.npy"):
    
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"File loaded: {file_name}")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embedding = model.encode([text], convert_to_numpy=True)
    print("Embedding shape:", embedding.shape)
    faiss.normalize_L2(embedding)
    dimension = embedding.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding.astype(np.float32))
    print("Vectors stored:", index.ntotal)
    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array([text]))
    print("FAISS index saved successfully.")
    tokenized_doc = [text.lower().split()]
    bm25 = BM25Okapi(tokenized_doc)
    print("BM25 index created.")
if __name__ == "__main__":
    build_index("my_document.txt")