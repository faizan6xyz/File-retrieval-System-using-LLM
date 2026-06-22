import numpy as np
import faiss
import os
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
def buildtext_vector(text , folder_path="../Data", chunk_path="../" , index_path="rag_index.faiss" , chunks_path="chunks.npy"):
    model = TextEmbedding("BAAI/bge-base-en-v1.5")  # same model, no hub needed
    new_embedding = np.array(list(model.embed([text])), dtype=np.float32)
    full_index_path  = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, chunks_path)
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







'''
1. PyMuPDF (fitz) - The "Swiss Army Knife"
    Best for: General-purpose text extraction, speed, and reliability.
    ✅ Fastest pure Python extractor for most PDFs
    ✅ Handles both text-based and some complex layouts
    ✅ Can extract images, metadata, and annotations
    ❌ Struggles with heavily formatted tables
    Use when: You need a reliable, fast default for 90% of PDFs.
    
        import fitz  # PyMuPDF
        def extract_text_pymupdf(pdf_path):
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
                if page_num < len(doc) - 1:
                    text += "\n"
            doc.close()
            return text
        text = extract_text_pymupdf("document.pdf")
        print(text[:]) 

2. pypdfium2 - The "Speed Demon"
    Best for: Processing thousands of PDFs or very large files.
    ✅ Extremely fast (uses C++ backend)
    ✅ Low memory footprint
    ❌ Less accurate with complex formatting
    ❌ No table extraction
    Use when: Speed is critical and layout precision is secondary.

        import pypdfium2 as pdfium
        def extract_text_pdfium(pdf_path):
            doc = pdfium.PdfDocument(pdf_path)
            text = ""
            for page in doc:
                text_page = page.get_textpage()
                text += text_page.get_text_bounded()
                text += "\n"
            doc.close()
            return text
        text = extract_text_pdfium("document.pdf")

3. pdfplumber - The "Layout Specialist"
    Best for: Financial reports, invoices, and documents with tables.
    ✅ Best-in-class table extraction
    ✅ Preserves spatial layout (x,y coordinates)
    ✅ Great for structured data extraction
    ❌ Slower than PyMuPDF
    Use when: You need to extract data from tables or preserve exact positioning.

        import pdfplumber
        def extract_text_pdfplumber(pdf_path):
            pdf = pdfplumber.open(pdf_path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n"
            pdf.close()
            return text
        text = extract_text_pdfplumber("document.pdf")

4. pymupdf4llm - The "RAG Optimizer"
    Best for: Preparing documents for LLMs and RAG systems.
    ✅ Converts PDFs directly to Markdown
    ✅ Preserves headers, lists, and code blocks
    ✅ Built on top of PyMuPDF but optimized for AI
    ❌ Newer, so fewer community examples
    Use when: You're building a RAG pipeline and want structure-aware chunks.

        import pymupdf4llm
        def extract_markdown_from_pdf(pdf_path):
            md_text = pymupdf4llm.to_markdown(pdf_path)
            return md_text
        markdown_text = extract_markdown_from_pdf("document.pdf")
        print(markdown_text[:1000])

5. Unstructured (local heavy lifter ) / LlamaParse (cloud heavy lifter)
    Best for: Production-grade, messy, or scanned documents.
    ✅ OCR built-in for scanned PDFs
    ✅ Handles mixed media (images + text)
    ✅ Auto-detects document type (invoice, article, etc.)
    ❌ Heavy dependencies, slower, often requires API keys
    Use when: You have no control over PDF quality or need enterprise-grade reliability.

        from unstructured.partition.pdf import partition_pdf
        def extract_with_unstructured(pdf_path):
            elements = partition_pdf(filename=pdf_path , strategy="hi_res")
            extracted_text = "\n".join([str(el) for el in elements])
            return extracted_text
        text = extract_with_unstructured("complex_document.pdf")
        print(text[:500])


        import nest_asyncio
        from llama_parse import LlamaParse
        async def extract_with_llamaparse(pdf_path, api_key):
            parser = LlamaParse( api_key=api_key , result_type="markdown" , verbose=True )
            documents = await parser.aload_data(pdf_path)
            return documents[0].text
        # Get key at: https://cloud.llamaindex.ai/
        api_key = "your_llama_cloud_api_key"
        text = nest_asyncio.run(extract_with_llamaparse("document.pdf", api_key))
        print(text[:500])

6. pdf2image + pytesseract - The "OCR Last Resort"
    Best for: Scanned PDFs where other libraries fail.
    ✅ Works on any image-based PDF
    ❌ Very slow
    ❌ Requires installing Tesseract OCR on Windows
    ❌ Lower accuracy than cloud-based OCR
    Use when: The PDF is purely images (scanned books, old archives)

        from pdf2image import convert_from_path
        import pytesseract
        def extract_text_ocr(pdf_path):
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image)
                text += "\n"
            return text

'''
"""
Comparison Table: Format Preservation

| Library          | Headers/Titles | Tables      | Columns     | Lists       | Output Format    |
| :---             | :---:          | :---:       | :---:       | :---:       | :---             |
| pymupdf4llm      | ✅ Excellent   | ✅ Good     | ✅ Good     | ✅ Good     | Markdown         |
| pdfplumber       | ❌ Manual      | ✅✅ Best   | ✅✅ Best   | ❌ Manual   | Raw Text + Coords|
| PyMuPDF (fitz)   | ❌ Lost        | ❌ Poor     | ❌ Jumbled  | ❌ Lost     | Raw String       |
| Unstructured     | ✅✅ Best      | ✅✅ Best   | ✅✅ Best   | ✅✅ Best   | JSON/Elements    |
"""

