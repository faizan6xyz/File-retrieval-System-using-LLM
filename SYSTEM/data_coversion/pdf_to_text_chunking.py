import pdfplumber
import numpy as np
import faiss
import os
import re
import pickle
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
os.environ["FASTEMBED_CACHE_PATH"] = r"C:\Users\faiza\.cache\huggingface\hub"
CHUNK_PATH  = "SYSTEM/RAG_data"
INDEX_PATH  = os.path.join(CHUNK_PATH, "rag_index.faiss")
CHUNKS_PATH = os.path.join(CHUNK_PATH, "chunks.npy")
META_PATH   = os.path.join(CHUNK_PATH, "metadata.npy")
BM25_PATH   = os.path.join(CHUNK_PATH, "bm25.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = TextEmbedding(model_name=MODEL_NAME, local_files_only=True)
def load_db():
    if not os.path.exists(INDEX_PATH):
        print("No existing database found. Fresh start.")
        return None, [], [], None
    index            = faiss.read_index(INDEX_PATH)
    existing_chunks  = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    existing_metadata = np.load(META_PATH, allow_pickle=True).tolist() \
                        if os.path.exists(META_PATH) else \
                        [{"source": "unknown", "chunk_id": i} for i in range(len(existing_chunks))]
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        print("Loaded bm25.pkl")
    else:
        print("bm25.pkl not found. Rebuilding from chunks...")
        tokenized = [re.findall(r"\w+", chunk.lower()) for chunk in existing_chunks]
        bm25 = BM25Okapi(tokenized)
    if index.ntotal != len(existing_chunks):
        raise ValueError(
            f"Mismatch: FAISS has {index.ntotal} vectors but chunks.npy has {len(existing_chunks)}. Rebuild index."
        )
    if len(existing_metadata) != len(existing_chunks):
        print("Warning: metadata count mismatch. Using placeholder metadata.")
        existing_metadata = [{"source": "unknown", "chunk_id": i} for i in range(len(existing_chunks))]
    print(f"Loaded rag_index.faiss : {index.ntotal} vectors")
    print(f"Loaded chunks.npy      : {len(existing_chunks)} chunks")
    print(f"Loaded metadata.npy    : {len(existing_metadata)} entries")
    return index, existing_chunks, existing_metadata, bm25
def save_db(index, all_chunks, all_metadata, bm25):
    os.makedirs(CHUNK_PATH, exist_ok=True)
    tmp_index    = INDEX_PATH.replace(".faiss", ".tmp")
    tmp_chunks   = CHUNKS_PATH.replace(".npy", ".tmp")
    tmp_metadata = META_PATH.replace(".npy", ".tmp")
    tmp_bm25     = BM25_PATH.replace(".pkl", ".tmp")
    faiss.write_index(index, tmp_index)
    np.save(tmp_chunks,   np.array(all_chunks,   dtype=object))
    np.save(tmp_metadata, np.array(all_metadata, dtype=object))
    with open(tmp_bm25, "wb") as f:
        pickle.dump(bm25, f)
    os.replace(tmp_index,             INDEX_PATH)
    os.replace(tmp_chunks   + ".npy", CHUNKS_PATH)
    os.replace(tmp_metadata + ".npy", META_PATH)
    os.replace(tmp_bm25,              BM25_PATH)
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
def semantic_chunking(sentences, threshold=0.75, max_chunk_size=20):
    if not sentences:
        return []
    print(f"Embedding {len(sentences)} sentences for semantic chunking...")
    embeddings = np.array(list(_model.embed(sentences)), dtype=np.float32)
    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        next_embedding = embeddings[i].reshape(1, -1)
        chunk_mean     = np.mean(
            embeddings[i - len(current_chunk):i], axis=0, keepdims=True
        )
        similarity = cosine_similarity(chunk_mean, next_embedding)[0][0]
        if similarity >= threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    print(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
    return chunks
def save_pdf(file_name, threshold=0.75, max_chunk_size=20,folder_path="SYSTEM/Data"):
    pdf_path = os.path.join(folder_path, file_name)
    if not os.path.exists(pdf_path):
        print(f" Error: File not found at {pdf_path}")
        return
    index, all_chunks, all_metadata, bm25 = load_db()
    print(f"Processing: {pdf_path}")
    text      = extract_text(pdf_path)
    sentences = split_into_sentences(text)
    chunks    = semantic_chunking(sentences, threshold=threshold, max_chunk_size=max_chunk_size)
    if not chunks:
        print("No chunks extracted from PDF.")
        return
    source_name = os.path.basename(pdf_path)
    start_id    = len(all_chunks)
    new_metadata = [
        {"source": source_name, "chunk_id": start_id + i}
        for i in range(len(chunks))
    ]
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = np.array(list(_model.embed(chunks)), dtype=np.float32)
    faiss.normalize_L2(embeddings)
    if index is None:
        dimension = embeddings.shape[1]
        index     = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    all_chunks.extend(chunks)
    all_metadata.extend(new_metadata)
    tokenized = [re.findall(r"\w+", chunk.lower()) for chunk in all_chunks]
    bm25      = BM25Okapi(tokenized)
    save_db(index, all_chunks, all_metadata, bm25)
    print(f"Total vectors in FAISS : {index.ntotal}")
    print(f"Total chunks           : {len(all_chunks)}")
    print(f"Total metadata entries : {len(all_metadata)}")
    print(f"{pdf_path} saved to database")
if __name__ == "__main__":
    save_pdf("h.pdf")
