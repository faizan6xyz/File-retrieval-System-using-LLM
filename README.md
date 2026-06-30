# 🗂️ File Retrieval System

A multimodal, AI-powered file retrieval and automation system for Windows that ingests PDFs, images, DOCX files, and videos — and lets you query your entire knowledge base through a local LLM.

---

## ✨ Features

- **Multimodal Ingestion** — indexes PDFs, images (with vision-model captioning), DOCX files, and video (frame extraction via OpenCV)
- **Hybrid Retrieval** — combines dense semantic search (fastembed + FAISS) with sparse keyword search (BM25) and fuses results with Reciprocal Rank Fusion (RRF)
- **Local-first** — runs entirely on your machine using Ollama or NVIDIA NIM; no cloud dependency
- **Vision Understanding** — uses a Qwen VLM (Qwen2.5-VL) to auto-generate text descriptions for images and video frames
- **Chat Interface** — query your indexed documents in a conversational loop backed by a local LLM
- **Safe Writes** — non-atomic write protection prevents index corruption on interrupted runs

---

## 🏗️ Architecture

```
Input Files (PDF / DOCX / Image / Video)
        │
        ▼
  ┌─────────────────────────────────────┐
  │         Ingestion Layer             │
  │  addfile.py  ─►  per-type chunkers  │
  │  video → frames (OpenCV)           │
  │  images → Qwen VLM captions        │
  └────────────────┬────────────────────┘
                   │  text chunks
                   ▼
  ┌─────────────────────────────────────┐
  │         Indexing Layer              │
  │  fastembed  ──►  FAISS (dense)     │
  │  BM25Okapi  ──►  BM25 index        │
  └────────────────┬────────────────────┘
                   │  at query time
                   ▼
  ┌─────────────────────────────────────┐
  │       Retrieval & Fusion            │
  │  RRF score fusion (0–1 normalized) │
  └────────────────┬────────────────────┘
                   │  top-k context
                   ▼
  ┌─────────────────────────────────────┐
  │         LLM Chat (local)            │
  │  Ollama / NVIDIA NIM (Llama 3.1)   │
  └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
File-retrieval-System-/
├── SYSTEM/
│   ├── addfile.py                  # Entry point — ingest a new file into the index
│   ├── image_to_text_chunking.py   # Vision pipeline: image → VLM caption → chunks
│   ├── video_to_image_byCV2.py     # Frame extraction from video using OpenCV
│   ├── chat.py                     # Conversational query interface
│   └── ...                         # Supporting modules (chunkers, retrieval, utils)
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10+
- Windows (primary target; Linux compatible with minor path adjustments)
- GPU: NVIDIA RTX (4 GB VRAM minimum for 4-bit quantized VLM)
- [Ollama](https://ollama.com/) **or** [NVIDIA NIM](https://build.nvidia.com/) for LLM inference

### Python dependencies

```bash
pip install fastembed faiss-cpu rank-bm25 python-docx pymupdf opencv-python Pillow
```

For vision captioning (Qwen VLM):

```bash
pip install transformers accelerate bitsandbytes
```

> **Note:** 4-bit quantization is required on 4 GB VRAM. The pipeline uses `load_in_4bit=True` via BitsAndBytes.

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/faizan6xyz/File-retrieval-System-.git
cd File-retrieval-System-
```

### 2. Set up your LLM backend

**Option A — Ollama**
```bash
ollama pull llama3.1:8b
```

**Option B — NVIDIA NIM**

Set your API key as an environment variable:
```bash
# .env
NIM_API_KEY=your_key_here
```

### 3. Index a file

```bash
python SYSTEM/addfile.py --file "path/to/your/document.pdf"
```

Supported formats: `.pdf`, `.docx`, `.png`, `.jpg`, `.mp4`, `.avi`

### 4. Chat with your indexed knowledge base

```bash
python SYSTEM/chat.py
```

---

## 🔍 How Retrieval Works

1. **Query** is embedded with `fastembed` (dense vector)
2. **BM25** scores all chunks for keyword overlap
3. **FAISS** returns top-k nearest neighbours by cosine similarity
4. **RRF fusion** normalizes both score lists into [0, 1] and combines them — the chunk that ranks well in *both* lists wins
5. Top fused chunks are passed as context to the local LLM

---

## 🧠 Vision Pipeline

For images and video:

1. Frames are extracted at a configurable FPS via OpenCV (`video_to_image_byCV2.py`)
2. Each frame/image is passed to **Qwen2.5-VL** running under 4-bit quantization
3. The generated caption is treated as a text chunk and indexed identically to document text

---

## 📌 Known Limitations

- FAISS dimension must stay consistent across all indexed files; re-index from scratch if you change the embedding model
- 4 GB VRAM constrains VLM batch size to 1 — captioning is sequential
- Video indexing is I/O-heavy; SSD recommended for large files

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 👤 Author

**Faizan** — [@faizan6xyz](https://github.com/faizan6xyz)

AI/ML Engineering Student · Freelance AI Engineer + System Desinger
