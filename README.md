# RAG Qwen PDF Chat Agent

An advanced **Retrieval-Augmented Generation (RAG)** system specifically optimized for analyzing Italian legal and notarized documents (contracts, deeds, leases). This project leverages **FastAPI** for the backend, **vLLM** for high-performance model inference, and **ChromaDB** as the vector store.

## Key Features

* **Legal-Specific Extraction**: Tailored system prompts designed to extract specific legal data like contract terms, parties involved, cadastral data, and payment details.
* **Smart PDF Processing**: Integrated OCR using `Tesseract` and `pdf2image` to process both native and scanned PDFs.
* **Sequential Map-Reduce Analysis**: Handles long documents by splitting text into manageable groups (max 24,000 characters) to ensure no critical information is lost during analysis.
* **Thinking Model Integration**: Utilizes "Thinking" LLMs (Qwen3-4B-Thinking) to provide detailed reasoning alongside answers.
* **Multi-User Support**: Dedicated session management and storage paths for different user IDs.

## Tech Stack

* **LLM Inference**: [vLLM](https://github.com/vllm-project/vllm).
* **Models**: 
    * **Chat**: `Qwen/Qwen3-4B-Thinking-2507`.
    * **Embeddings**: `Qwen/Qwen3-Embedding-0.6B`.
* **Framework**: LangChain (Core, Classic, Community).
* **Vector Database**: ChromaDB.
* **API Framework**: FastAPI.
* **OCR**: Tesseract OCR (with Italian language support).

## Setup & Installation

### Prerequisites
* Docker and Docker Compose
* NVIDIA GPU (Recommended for vLLM performance)
* Hugging Face API Token

### Configuration
1.  Create a `.env` file in the root directory:
    ```env
    HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
    API_KEY=your_secret_api_key
    ```

2.  Launch the services:
    ```bash
    docker-compose up -p --build
    ```
    * **FastAPI Backend**: `http://localhost:8000`
    * **vLLM Server**: `http://localhost:8001`

## API Endpoints

All requests require the `X-API-Key` header for authentication.

### 1. Upload PDF
* **Endpoint**: `POST /upload_pdf`
* **Header**: `X-User-Id: <unique_user_id>`
* **Payload**: `multipart/form-data` with a `files` field.
* **Description**: Uploads the PDF, indexes it in the vector store, and returns an automated standard legal analysis.

### 2. Chat
* **Endpoint**: `POST /chat`
* **Header**: `X-User-Id: <unique_user_id>`
* **Payload**: 
    ```json
    { "question": "What is the duration of the lease?" }
    ```
* **Response**: Returns the `answer` and the model's `reasoning` process.

## Project Structure

* `app/`: FastAPI application logic, routers, and schemas.
* `src/`: Core RAG engine logic.
    * `ingestion.py`: Document splitting and vector embedding.
    * `smart_pdf_loader.py`: PDF text extraction and OCR fallback.
    * `pdf_chat.py`: Retrieval chain and chat logic.
* `Dockerfile`: Environment setup including Tesseract and system dependencies.
* `docker-compose.yml`: Multi-container orchestration.

## ⚖️ Disclaimer
This tool is intended for assistive purposes in legal document analysis. Always verify the extracted information with a qualified legal professional.