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
model = SentenceTransformer("BAAI/bge-base-en-v1.5") # Load the BGE model for generating embeddings, which captures semantic meaning of text
embeddings = model.encode(all_chunks, convert_to_numpy=True) # Generates a 768-dim vector for each chunk capturing its semantic meaning
print("Embedding shape:", embeddings.shape) # give the shape of the embeddings array, which should be (number_of_chunks, embedding_dimension)
'''
faiss.normalize_L2(embeddings) # Normalize the embeddings to ensure that cosine similarity can be computed as an inner product in the FAISS index, which is essential for accurate similarity search based on the angle between vectors rather than their magnitude.
dimension = embeddings.shape[1] # get the value of embedding dimension
index = faiss.IndexFlatIP(dimension) # IndexFlatIP = exact brute-force search using inner product. Since embeddings are L2-normalized, inner product == cosine similarity
index.add(embeddings.astype(np.float32)) # Store all embeddings inside that database for fast similarity search.
print("Vectors stored:", index.ntotal) 
faiss.write_index(index, "rag_index.faiss") # Save the FAISS index to disk, allowing for persistent storage of the vector database and enabling efficient retrieval of relevant chunks based on query embeddings without needing to recompute the index every time the program runs.
np.save("chunks.npy", np.array(all_chunks)) # Save the original text chunks to a NumPy file, ensuring that we have access to the actual content corresponding to each embedding in the FAISS index for retrieval and display purposes. This allows us to easily load the chunks later when we need to retrieve and display them based on user queries.
print("FAISS index saved successfully.")
'''
index = faiss.read_index("rag_index.faiss") # Load the FAISS index from disk, allowing for efficient retrieval of relevant chunks based on query embeddings. This step is necessary to perform similarity search without having to recompute embeddings every time the program runs.
all_chunks = np.load("chunks.npy", allow_pickle=True) # Load the original text chunks from disk, ensuring that we have access to the actual content corresponding to each embedding in the FAISS index for retrieval and display purposes.
from rank_bm25 import BM25Okapi
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]  # create a list of tokenized chunks for BM25
bm25 = BM25Okapi(tokenized_chunks) # Initialize BM25 with the tokenized chunks, allowing for keyword-based search
print("BM25 index created.")
def retrieve(query, k=3):
    query_embedding = model.encode("Represent this sentence for searching relevant passages: " + query, convert_to_numpy=True) # Generate an embedding for the query, which will be used to perform a similarity search in the FAISS index. The query is prefixed with a prompt to guide the embedding model to produce a representation that is suitable for retrieving relevant passages based on semantic similarity.
    query_embedding = np.array([query_embedding], dtype=np.float32) # Reshape the query embedding to match the expected input shape for FAISS search.
    faiss.normalize_L2(query_embedding) # Normalize the query embedding to ensure that cosine similarity is equivalent to inner product search in FAISS.
    dense_scores, dense_indices = index.search(query_embedding, k=len(all_chunks)) # Perform a similarity search in the FAISS index using the query embedding, retrieving scores and indices for all chunks. The scores represent the similarity between the query and each chunk, while the indices indicate which chunks correspond to those scores.
    dense_scores = dense_scores[0] # Extract the scores from the search results, which are returned as a 2D array. Since we only have one query, we take the first row to get a 1D array of scores corresponding to each chunk in the index.
    dense_indices = dense_indices[0] # Extract the indices from the search results, which are returned as a 2D array. Similar to scores, we take the first row to get a 1D array of indices corresponding to each chunk in the index.
    tokenized_query = query.lower().split() # Tokenize the query for BM25 keyword search.
    bm25_scores = bm25.get_scores(tokenized_query)  # Get BM25 scores for each chunk based on keyword matching with the tokenized query.
    aligned_dense_scores = np.zeros(len(all_chunks))
    for score, idx in zip(dense_scores, dense_indices):
        aligned_dense_scores[idx] = score # FAISS returns scores in rank order, not chunk order. This loop re-maps each score back to its original chunk position so it can be added element-wise with bm25_scores
    dense_scores_norm = (aligned_dense_scores - aligned_dense_scores.min()) / \
                        (aligned_dense_scores.max() - aligned_dense_scores.min() + 1e-8)     # Normalize the dense scores to a 0-1 range for fair combination with BM25 scores.
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)   # Normalize the BM25 scores to a 0-1 range for fair combination with dense scores.
    final_scores = (0.7 * dense_scores_norm + 0.3 * bm25_scores_norm)  # Combine the normalized scores with weights (0.7 for dense and 0.3 * BM25) to create a hybrid relevance score.
    top_indices = np.argsort(final_scores)[::-1][:k] # Get the indices of the top k chunks based on the final combined scores, sorting them in descending order to retrieve the most relevant chunks according to the hybrid scoring mechanism.
    for rank, idx in enumerate(top_indices):
        print(f"Rank {rank+1}")
        print(f"Hybrid Score: {final_scores[idx]:.4f}")
        print(all_chunks[idx])
        print("-" * 60)
    return [all_chunks[idx] for idx in top_indices] # Return the original text chunks corresponding to the top indices as retrieved results.
if __name__ == "__main__": # This block ensures that the retrieval function is only executed when this script is run directly, allowing for modularity and preventing unintended execution when imported as a module in other scripts.
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print("\nRetrieving relevant chunks...\n")
        retrieve(query) # Call the retrieve function to get and display the most relevant chunks based on the user's query, combining both semantic similarity and keyword matching for improved retrieval performance.