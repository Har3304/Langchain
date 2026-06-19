# Simple RAG Pipeline with LangChain and Llama-3.1

This repository demonstrates a minimal, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built using LangChain and Hugging Face. The system indexes private domain documents, retrieves context relevant to a user's query, and passes that context to `Llama-3.1-8B-Instruct` to generate an accurate, factual answer.

## Workflow Architecture

1. **Phase 1: Indexing** - Raw data documents are loaded.
   - Text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
   - Chunks are converted into vector embeddings via `HuggingFaceEmbeddings` and stored in a local `FAISS` vector database.
2. **Phase 2: Querying (Retrieval)**
   - The user query is embedded, and the vector store fetches the top $K=2$ most semantically relevant text chunks.
3. **Phase 3: Generation (Augmentation)**
   - The retrieved context chunks are formatted into a string and structured cleanly into a prompt template alongside the user's question.
   - The final augmented prompt is sent to `Llama-3.1-8B-Instruct` for zero-hallucination inference.

---

## Prerequisites

### 1. Hugging Face Access Token
You will need a Hugging Face User Access Token with **Read** permissions to download models and use the Hugging Face Inference Endpoint. 
* Generate a token in your [Hugging Face Settings](https://huggingface.co/settings/tokens).
* If you are running this code inside **Google Colab**, navigate to the **Secrets** tab on the left sidebar and add a new secret key named `HF_TOKEN_READ` with your token value.

### 2. Model Access
Ensure you have accepted the community license agreement for `meta-llama/Llama-3.1-8B-Instruct` on the Hugging Face Hub if required.

---

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
