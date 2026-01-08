# RAG PDF Chatbot with Ollama

This repository provides a solution for uploading private documents (PDFs) and interacting with them using a Retrieval-Augmented Generation (RAG) pipeline. 
With this project, you can ask questions about your uploaded documents and summarize content.

It leverages **Chroma** for vector storage, **Ollama's embedding**, and **Ollama's LLaMA** model for chat and question-answering.

---

## Features

- **Upload Private Documents**: Securely upload and process your own PDFs.
- **Question Answering**: Ask questions about the content in your documents.
- **Summarization**: Generate summaries for your uploaded documents.
- **Vector Storage with Chroma**: Document ingestion and storage.
- **LLaMA-Powered Chat**: Leverages Ollama's LLaMA models.

---

## How It Works

1. **Document Ingestion**:  
   Uploaded PDFs are processed, and text is extracted.
   
2. **Embedding**:  
   Text data is embedded into vector representations using a pre-trained model.

3. **Vector Store with Chroma**:  
   The embeddings are stored in Chroma, enabling efficient retrieval.

4. **Chat and Summarization**:  
   Using Ollama's LLaMA, the system responds to queries or generates summaries based on the document content.

---

## Usage

### Ensure to choose a model that suits your hardware capacity, I suggest:

- Chat model **llama3.1:8b** 
- Embedding model **nomic-embed-text** 

### Getting started

1. **Upload a PDF file.**
2. **Run main.py after setting custom parameters**
3. **Ask questions or request a summary of the content.**

  #### For example:

    "What is the main idea of this document?"
    
    "Summarize the document."

