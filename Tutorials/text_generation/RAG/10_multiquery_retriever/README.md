# Advanced RAG with MultiQueryRetriever and Hugging Face

This repository demonstrates a production-style Retrieval-Augmented Generation (RAG) pipeline built with **LangChain (LCEL)**, **Hugging Face**, and **ChromaDB**. 

It implements a **MultiQueryRetriever**, which automates the process of prompt tuning by using an LLM to generate multiple versions of a user's query from different perspectives. This overcomes the limitations of traditional distance-based vector searches by retrieving a richer, more comprehensive set of context documents.

## Features

* **Multi-Query Generation:** Uses `meta-llama/Meta-Llama-3-8B-Instruct` via Hugging Face Endpoints to split a single user query into multiple variations.
* **Vector Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for highly efficient, local semantic embeddings.
* **Vector Store:** Leverages `Chroma` for rapid, in-memory document storage and similarity retrieval.
* **LangChain Expression Language (LCEL):** Features a clean, declarative chain mapping context and questions seamlessly to the LLM.

---

## Setup Instructions

### 1. Clone and Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install required packages:

```bash
pip install -r requirements.txt
