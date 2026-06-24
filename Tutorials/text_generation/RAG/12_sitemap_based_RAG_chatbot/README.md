# Granular Deduplicating Sitemap Crawler & RAG Ingestion Pipeline

An advanced, token-efficient web scraping and data ingestion pipeline built for Retrieval-Augmented Generation (RAG) workflows. This system processes entire websites via their XML sitemaps, segments pages into modular HTML categories, drops duplicate boilerplate text (like headers, footers, and global menus) at an atomic line level, and structures novel content via LLMs directly into an vector database.

## Key Features

* **Recursive Sitemap Discovery:** Seamlessly parses standard XML sitemaps as well as nested sitemap indexes.
* **Structural Semantic Segmentation:** Divides raw HTML payload into three isolated logical spaces before processing: `metadata`, `navigation`, and `content`.
* **Line-Level Atomic Deduplication:** Features a stateful fingerprint engine tracking text chunks via MD5 hashes across the entire crawl session. Avoids feeding repeated text blocks (e.g., global navigation links, unified footers) into the LLM multiple times, saving significant context tokens.
* **Context-Preserving Heading Anchors:** Intelligent buffer groupings prevent structural headlines (`<h1>`, `<h2>`, `<h3>`) from becoming disconnected from their sibling paragraphs during deduplication passes.
* **LLM Schema Structuring:** Drives `meta-llama/Meta-Llama-3-8B-Instruct` through LangChain to normalize raw chunks into clean, markdown-mapped schemas.
* **Native Vector RAG Storage:** Ingests cleaned data profiles automatically into a local `Chroma` database backed by `sentence-transformers/all-MiniLM-L6-v2` embeddings.

---

## Installation

1. Clone or download this project workspace.
2. Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
