import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
folder_path = "Data"
all_chunks = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                all_chunks.append(paragraph)
print(f"Total chunks: {len(all_chunks)}")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(all_chunks,convert_to_numpy=True)
print("Embedding shape:", embeddings.shape)
faiss.normalize_L2(embeddings) # give (10 number of the text chunk , 786 size of each embedding vector )
dimension = embeddings.shape[1] # get the value of dimension
index = faiss.IndexFlatIP(dimension) # this Create an empty FAISS database.
index.add(embeddings.astype(np.float32))# Store all embeddings inside that database for fast similarity search.
print("Vectors stored:", index.ntotal) 
faiss.write_index(index, "rag_index.faiss") # Stores the vector embeddings
np.save("chunks.npy", np.array(all_chunks)) # Stores the original paragraph text corresponding to each vector
print("FAISS index saved successfully.")
# Load FAISS index , index = faiss.read_index("rag_index.faiss") 
# Load text chunks , all_chunks = np.load("chunks.npy", allow_pickle=True)
while True:
    query = input("\nEnter your query (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    query_embedding = model.encode(query,convert_to_numpy=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k=3)
    print("\nTop matching chunks:\n")
    for rank, idx in enumerate(indices[0]):
        print(f"Rank {rank + 1}")
        print(f"Similarity Score: {scores[0][rank]:.4f}")
        print(all_chunks[idx])
        print("-" * 60)