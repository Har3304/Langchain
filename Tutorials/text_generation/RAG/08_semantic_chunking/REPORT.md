# RAG Chunking & Retrieval Benchmarking Report

## 1. Executive Summary
This report evaluates the performance of two prominent text-splitting strategies—**Recursive Character Splitting** and **Semantic Chunking**—using a standardized benchmark dataset focused on Enterprise Resource Planning (ERP) corporate domain knowledge. 

A standard text block defining an ERP "Company Code" was evaluated against a suite of six targeted business and operational queries using a **Chroma vector database** paired with **Hugging Face Embeddings**.

---

## 2. Experimental Setup & Configuration

### Data Input
The test asset consists of four distinct paragraphs detailing the definition, accounting principles, modular integration, and governance requirements of an ERP company code.

### Segmentation Configurations
1. **Recursive Character Text Splitter**
   - **Chunk Size:** 700 characters
   - **Chunk Overlap:** 100 characters
   - **Separators:** `['\\n\\n', '\\n', ' ', '']`
2. **Semantic Chunker**
   - **Embedding Engine:** HuggingFaceEmbeddings
   - **Breakpoint Type:** Percentile
   - **Threshold:** 90th percentile

---

## 3. Comparative Analysis: Chunking Characteristics

The structural differences between programmatic character-bound constraints and semantic distance constraints resulted in distinct data geometries:

| Metric / Attribute | Recursive Character Splitter | Semantic Chunker (90th Percentile) |
| :--- | :--- | :--- |
| **Total Chunks Generated** | 4 chunks | 3 chunks |
| **Granularity / Size Variance** | **Uniform:** Chunks range consistently between 452 and 565 characters. | **Variable:** Chunks range dramatically from 341 characters to 877 characters. |
| **Boundary Logic** | Relies on structural delimiters (`\\n\\n`) to fulfill mathematical length targets. | Evaluates the mathematical distance between consecutive sentence embeddings. |

### Semantic Splitting Behavior
Because the `breakpoint_threshold_amount` was configured to a high value (**90**), the semantic model required a dramatic shift in vector distance to trigger a separation boundary. As a result, the engine concatenated several distinct sub-topics (integration and governance) into an expansive, single 877-character block (Chunk 3).

---

## 4. Critical Bug Discovery: Database Duplication
Review of the raw output reveals an artifact where the top-ranked results contain duplicate entries:
