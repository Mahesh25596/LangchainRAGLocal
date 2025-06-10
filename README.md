# LangchainRAGLocal - Local Document Search with LangChain and ChromaDB

A Python-based document search system that uses HuggingFace embeddings and ChromaDB for efficient local document query retrieval.

## Features

- 📂 Load and process markdown documents from a directory
- ✂️ Intelligent text splitting with configurable chunk sizes
- 🔍 Semantic search using HuggingFace's `all-MiniLM-L6-v2` embeddings
- 💾 Persistent vector storage with ChromaDB
- 🖥️ Command-line query interface

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/document-search.git
   cd document-search

2. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Create database
    ```bash
    python create_database.py
    ```
4. Query the database

    ```bash
    python query_data.py "List AWS Skillsets"
    ```