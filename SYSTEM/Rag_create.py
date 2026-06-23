import os
import re
import numpy as np
import faiss
import pickle
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
os.environ["FASTEMBED_CACHE_PATH"] = r"C:\Users\faiza\fastembed_models"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE_TOKENS = 400      # Hard cap — safety net if a topic runs long
MIN_CHUNK_TOKENS = 50        # Merge chunks smaller than this into a neighbor
BATCH_SIZE = 32              # Process embeddings in batches to save RAM
PERCENTILE_THRESHOLD = 25    # Lower percentile of similarity drops = cut point
_model = TextEmbedding(MODEL_NAME)
def estimate_tokens(text: str) -> float:
    return sum(1 + len(w) / 4 for w in text.split())
def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]
def chunk_text_semantic(text,chunk_size=CHUNK_SIZE_TOKENS,min_chunk_tokens=MIN_CHUNK_TOKENS,percentile=PERCENTILE_THRESHOLD):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text else []
    sentence_embeddings = np.stack(list(_model.embed(sentences))).astype(np.float32)
    faiss.normalize_L2(sentence_embeddings)
    sims = np.array([
        float(np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]))
        for i in range(len(sentence_embeddings) - 1)
    ])
    threshold = np.percentile(sims, percentile)
    cut_after = set(i for i, s in enumerate(sims) if s <= threshold)
    raw_chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if (i - 1) in cut_after:
            raw_chunks.append(current)
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    raw_chunks.append(current)
    merged_chunks = []
    buffer = []
    buffer_tokens = 0
    for chunk_sentences in raw_chunks:
        chunk_text_str = " ".join(chunk_sentences)
        chunk_tokens = estimate_tokens(chunk_text_str)
        buffer.extend(chunk_sentences)
        buffer_tokens += chunk_tokens
        if buffer_tokens >= min_chunk_tokens:
            merged_chunks.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0
    if buffer:  # leftover tail too small — attach to last chunk if possible
        if merged_chunks:
            merged_chunks[-1] = merged_chunks[-1] + " " + " ".join(buffer)
        else:
            merged_chunks.append(" ".join(buffer))
    final_chunks = []
    for chunk in merged_chunks:
        if estimate_tokens(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            words = chunk.split()
            sub_start = 0
            while sub_start < len(words):
                sub_chunk_words = []
                token_count = 0
                idx = sub_start
                while idx < len(words):
                    t = 1 + len(words[idx]) / 4
                    if token_count + t > chunk_size:
                        break
                    token_count += t
                    sub_chunk_words.append(words[idx])
                    idx += 1
                if not sub_chunk_words:  # single huge word edge case
                    sub_chunk_words = [words[sub_start]]
                    idx = sub_start + 1
                final_chunks.append(" ".join(sub_chunk_words))
                sub_start = idx
    return final_chunks
def load_existing_data(chunk_path, index_path):
    full_index_path = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, "chunks.npy")
    full_meta_path = os.path.join(chunk_path, "metadata.npy")
    index = None
    existing_chunks = []
    existing_metadata = []
    dimension = 768
    if os.path.exists(full_index_path):
        index = faiss.read_index(full_index_path)
        dimension = index.d
        existing_chunks = np.load(full_chunks_path, allow_pickle=True).tolist()
        if os.path.exists(full_meta_path):
            existing_metadata = np.load(full_meta_path, allow_pickle=True).tolist()
        print(f"Loaded existing index with {index.ntotal} vectors.")
    else:
        print("No existing index found. Creating new one.")
    return index, existing_chunks, existing_metadata, dimension
def save_data(index, chunks, metadata, chunk_path, index_path, bm25):
    full_index_path = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, "chunks.npy")
    full_meta_path = os.path.join(chunk_path, "metadata.npy")
    full_bm25_path = os.path.join(chunk_path, "bm25.pkl")
    faiss.write_index(index, full_index_path + ".tmp")
    os.replace(full_index_path + ".tmp", full_index_path)
    tmp_chunks_path = full_chunks_path.replace(".npy", ".tmp.npy")
    tmp_meta_path = full_meta_path.replace(".npy", ".tmp.npy")
    np.save(tmp_chunks_path, np.array(chunks, dtype=object))
    np.save(tmp_meta_path, np.array(metadata, dtype=object))
    os.replace(tmp_chunks_path, full_chunks_path)
    os.replace(tmp_meta_path, full_meta_path)
    with open(full_bm25_path + ".tmp", 'wb') as f:
        pickle.dump(bm25, f)
    os.replace(full_bm25_path + ".tmp", full_bm25_path)
    print("All data saved successfully.")
def build_index(file_name,
                 folder_path="SYSTEM/Data",
                 chunk_path="SYSTEM/RAG_data",
                 index_path="rag_index.faiss"):
    os.makedirs(chunk_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    index, existing_chunks, existing_metadata, dimension = load_existing_data(
        chunk_path, index_path
    )
    if index is None:
        index = faiss.IndexFlatIP(dimension)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    new_chunks = chunk_text_semantic(text)
    print(f"Generated {len(new_chunks)} new chunks from {file_name}")
    existing_set = set(existing_chunks)
    unique_new_chunks = []
    unique_new_metadata = []
    for chunk in new_chunks:
        if chunk not in existing_set:
            chunk_id = len(existing_chunks) + len(unique_new_chunks)
            unique_new_chunks.append(chunk)
            unique_new_metadata.append({
                "source": file_name,
                "chunk_id": chunk_id
            })
    if not unique_new_chunks:
        print("No new unique chunks to add.")
        return
    print(f"Adding {len(unique_new_chunks)} unique chunks after deduplication.")
    all_embeddings = []
    for i in range(0, len(unique_new_chunks), BATCH_SIZE):
        batch = unique_new_chunks[i:i + BATCH_SIZE]
        batch_embs = np.stack(list(_model.embed(batch))).astype(np.float32)
        faiss.normalize_L2(batch_embs)
        all_embeddings.append(batch_embs)
    if not all_embeddings:
        return
    new_embeddings = np.vstack(all_embeddings)
    index.add(new_embeddings)
    final_chunks = existing_chunks + unique_new_chunks
    final_metadata = existing_metadata + unique_new_metadata
    print("Rebuilding BM25 index...")
    tokenized_corpus = [chunk.lower().split() for chunk in final_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    save_data(index, final_chunks, final_metadata, chunk_path, index_path, bm25)
    print(f"Indexing complete. Total vectors: {index.ntotal}")
if __name__ == "__main__" :
    build_index("texting.txt")