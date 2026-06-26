# Semantic Cache with Multi-Query Retrieval Expansion

A high-efficiency semantic caching framework built using **LangChain** and **Hugging Face Hub**. 

This system uses a `MultiQueryRetriever` to expand an incoming user query into several linguistic variations. It matches these variations against an in-memory vector cache using a strict **>95% Cosine Similarity barrier**, significantly reducing LLM pricing footprints and lowering response latencies.

## System Architecture

1. **Query Inflation**: The incoming query is split into multiple semantic variants via a fast base-model prompt wrapper (`meta-llama/Llama-3.1-8B`).
2. **Vector Space Lookup**: All variations are parallelized against an internal dictionary stored as dense numerical embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
3. **Threshold Gate**: If any variant matches a past cache entry at $\ge 0.95$ cosine similarity, the text response is served immediately.
4. **Fallback Execution**: On a total cache miss, the primary LLM is called, and the raw text output is mapped against the original vector coordinate to seed future lookups.

## Getting Started

### 1. Prerequisites
Clone or download this directory to your machine and ensure Python 3.9+ is active.

### 2. Install Dependencies
Install all required libraries through pip:
```bash
pip install -r requirements.txt
```

### 3. Provide Credentials
You need a Hugging Face token authorized to fetch model gates. Set it in your terminal environment:

**On Linux/macOS:**
```bash
export HF_TOKEN="your_huggingface_read_token"
```

**On Windows (Command Prompt):**
```cmd
set HF_TOKEN="your_huggingface_read_token"
```

### 4. Run the Demo Routine
Execute the built-in scenario sequence to observe multi-query cache hits:
```bash
python demo.py
```

## Performance Characteristics
* **Time to First Token (Cache Hit)**: ~0.02s (Local vector math bound)
* **Time to First Token (Cache Miss)**: ~1.5s - 4.0s (Network/Inference bound)
