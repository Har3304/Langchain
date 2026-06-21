# LangSmith Tracing Demo with HuggingFace

This repository demonstrates how to implement end-to-end tracing in LangSmith using LangChain and Hugging Face models (`meta-llama/Llama-3.1-8B-Instruct`). It features forced trace flushing to ensure all execution logs arrive safely at your LangSmith dashboard when running in short-lived environments like Google Colab or minimal scripts.

## Features
* **Modern Configuration:** Uses up-to-date `LANGSMITH_*` environment variable layouts alongside legacy `LANGCHAIN_*` fallbacks.
* **Targeted Tracking:** Uses the `@traceable` decorator to cleanly isolate and tag production summarization pipelines.
* **Synchronous Flushing:** Employs the LangSmith `Client` queue flush at execution termination to prevent dropped logs in notebook runtimes.

## Prerequisites

Before running the script, ensure you have gathered the following credentials:
1. **Hugging Face Token:** A token with read access to repository assets (requested from your Hugging Face Account settings).
2. **LangSmith API Key:** Your private API key generated from your LangSmith profile workspace dashboard.

## Setup and Installation

1. Clone or copy the project script to your environment.
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
