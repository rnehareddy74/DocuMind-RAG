# DocuMind RAG — Intelligent Multi-Document Q&A System

An advanced Retrieval-Augmented Generation (RAG) pipeline that transforms complex, multi-document PDFs into an interactive Q&A experience. Built with Mistral-7B, FAISS vector search, and a clean Gradio interface.

---

## Overview

DocuMind RAG goes beyond simple PDF chat. When you upload a PDF, the system intelligently detects document boundaries, classifies each logical document by type (Invoice, Contract, Resume, Bank Statement, etc.), and routes your questions to the most relevant document automatically.

---

## Features

- **Automatic Document Segmentation** — Detects and separates multiple logical documents within a single PDF (e.g., a packet containing a Pay Slip, a Bank Statement, and a Contract)
- **LLM-Powered Classification** — Uses Mistral-7B to classify each document into one of 16 categories
- **Smart Query Routing** — Predicts which document type is most likely to contain the answer and retrieves from that sub-index first
- **OCR Fallback** — Automatically applies Tesseract OCR on scanned or image-based pages
- **FAISS Vector Search** — Builds per-document-type FAISS indices for fast, filtered retrieval
- **Source Attribution** — Every answer cites the document type and page range it drew from
- **Gradio Web UI** — Upload, process, and chat through a browser-based interface with document type filtering controls

---

## Architecture

```
PDF Upload
    │
    ▼
PDF Extraction (PyMuPDF + OCR fallback)
    │
    ▼
Page-by-Page Analysis
    ├── Document Boundary Detection  (LLM)
    └── Document Type Classification (LLM)
    │
    ▼
Logical Document Segmentation
    │
    ▼
Chunking (LlamaIndex SentenceSplitter, 500 tokens, 100 overlap)
    │
    ▼
Embedding (all-MiniLM-L6-v2)
    │
    ▼
FAISS Index (global + per-document-type sub-indices)
    │
    ▼
Query → Route → Retrieve → Generate Answer (Mistral-7B)
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended; CPU also supported)
- Tesseract OCR installed on the system

### Python Packages

```
gradio
gradio_pdf
pypdf
PyPDF2
pymupdf
sentence-transformers
transformers
faiss-cpu
torch
numpy
pandas
llama-index
llama-index-readers-file
llama-index-embeddings-huggingface
llama-index-vector-stores-faiss
llama-cpp-python==0.3.0
pillow
pytesseract
```

---

## Quick Start (Google Colab)

The project is designed to run in Google Colab with a GPU runtime.

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1FNPMjPdFe3cyFsxJjrXXlu3VEfJYD0jh)
2. Set runtime to **GPU** (T4 or better recommended)
3. Add your Hugging Face token to Colab Secrets as `Huggingface`
4. Run all cells — the Mistral-7B GGUF model downloads automatically
5. A public Gradio link is printed at the end

---

## Usage

### Upload & Process

1. Upload a PDF using the file input panel
2. Click **Process Document** — the system extracts, segments, and indexes automatically
3. The Document Info panel shows how many logical documents were found and their types

### Ask Questions

- Type any question in the chat box and press **Send**
- Use the **Document Type Filter** dropdown to restrict retrieval to a specific document type
- Toggle **Auto-Route Queries** to let the LLM predict the best document type automatically
- Adjust **Chunks to Retrieve** (1–10) to control context window size

### Quick Actions

- **What's the summary?** — Generates a summary of the main points
- **Find amounts** — Extracts all monetary figures and financial values

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_ctx` | 4096 (LLM) / 1024 (store) | Context window size |
| `chunk_size` | 500 tokens | Size of each text chunk |
| `chunk_overlap` | 100 tokens | Overlap between consecutive chunks |
| `k` (retrieval) | 4 | Number of chunks to retrieve per query |
| `temperature` | 0.2 | LLM generation temperature |
| `routing_confidence` | 0.7 | Minimum confidence to use type-specific index |

---

## Supported Document Types

Resume, Contract, Mortgage Contract, Invoice, Pay Slip, Lender Fee Sheet, Land Deed, Bank Statement, Tax Document, Insurance, Report, Letter, Form, ID Document, Medical, Other

---

## Project Structure

```
enhanced_document_qa_rag.py     # Full pipeline (single-file Colab notebook export)
  ├── Data Structures             # PageInfo, LogicalDocument, ChunkMetadata
  ├── PDF Extraction              # PyMuPDF + Tesseract OCR
  ├── Document Intelligence       # Classification & boundary detection (LLM)
  ├── Chunking                    # Word-level + LlamaIndex SentenceSplitter
  ├── IntelligentRetriever        # FAISS indices + query routing
  ├── Answer Generation           # Source-attributed generation with Mistral-7B
  ├── EnhancedDocumentStore       # Orchestration class
  └── Gradio Interface            # Web UI
```

---

## Limitations

- Designed for Google Colab; running locally requires adjusting the model download path and removing Colab-specific imports (`google.colab.userdata`)
- Very large PDFs (100+ pages) may take several minutes to process due to per-page LLM calls for classification and boundary detection
- Model accuracy depends on Mistral-7B-Instruct-v0.2 Q4 quantization quality

---

## License

This project is for educational and research use. The Mistral-7B model weights are subject to the [Mistral AI License](https://mistral.ai/licenses/MRL-0.1.md).
