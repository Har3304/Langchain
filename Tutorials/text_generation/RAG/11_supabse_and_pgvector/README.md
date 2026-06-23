
# Supabase PGVector Connection Verifier

This repository contains a simple utility script written in Python to verify a connection to a Supabase PostgreSQL instance using the `PGVector` extension via LangChain. It uses a Hugging Face embedding model to generate embeddings, saves a test document to the vector store, performs a semantic similarity search, and then cleans up by deleting the test document.

## Features
- **Hugging Face Integration**: Automatically generates vector embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model.
- **Supabase PGVector Store**: Connects directly to a Supabase PostgreSQL instance utilizing `pgvector`.
- **Validation Pipeline**: Full integration test that verifies insertion, similarity search capability, and deletion permissions.

## Prerequisites
Before running the script, ensure you have:
1. A **Supabase Project** with the `pgvector` extension enabled.
2. A database connection string (`SUPABASE_DATABASE_URL`).
3. A **Hugging Face Access Token** (Read permission).

## Installation

1. **Clone or copy the script** to your local environment or Google Colab notebook.
2. **Install the required packages** using `pip`:
   ```bash
   pip install -r requirements.txt
