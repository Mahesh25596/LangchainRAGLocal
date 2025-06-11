# LangchainRAGLocal - Local RAG System with Ollama and ChromaDB

A complete local Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and Ollama for private document search and question answering.

## Features

- - 📂 **Document Processing**  
  Load and process PDF/markdown documents with configurable text splitting
- ✂️ **Chunk Optimization**  
  Intelligent text splitting with overlap for context preservation
- 🔍 **Semantic Search**  
  HuggingFace `all-MiniLM-L6-v2` embeddings for accurate retrieval
- 🤖 **Local LLM Integration**  
  Ollama-powered responses with models like LLaMA2/Mistral
- 💾 **Persistent Storage**  
  ChromaDB vector store for efficient document retrieval
- 🖥️ **CLI Interface**  
  Simple command-line query system with output redirection

## Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for larger models)
- Ollama installed (for local LLM)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mahesh25596/LangchainRAGLocal.git
   cd LangchainRAGLocal
   ```

2. Install Dependencies

    ```bash
    winget install Ollama.Ollama
    ollama pull llama2
    pip install -r requirements.txt
    ```

3. Create database
    ```bash
    python create_database.py
    ```
4. Query the database

    ```bash
    python query_data.py "List AWS Skillsets" > output.txt
    ```