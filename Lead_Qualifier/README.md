# Lead Qualifier

A structured data extraction pipeline built with **LangChain (LCEL)** and **Pydantic**, powered by open-source LLMs hosted via Hugging Face Inference Endpoints. 

This repository parses raw, unstructured descriptions of corporate entities to extract key data points and evaluate them against custom B2B lead qualification criteria automatically.

## Features
- **Deterministic Extraction:** Structured data schema validation enforced natively using Pydantic templates.
- **LCEL Architecture:** Built using modern LangChain Expression Language pipelines for flexible streaming and scaling.
- **Open-Source Ready:** Fully integrated via `langchain-huggingface` to leverage state-of-the-art open models like Llama 3.1 or Mistral without requiring OpenAI API keys.

## Schema Architecture
The pipeline enforces a strict structured data model mapping the following parameters from raw corporate copy:
- `company_name` (str): The formal name of the target service provider.
- `industry` (str): The primary operations horizontal or vertical vertical.
- `is_qualified` (bool): True if the company matches defined B2B operational profiles.

## Setup Instructions

### 1. Clone & Navigate
```bash
git clone [https://github.com/Har3304/Langchain.git](https://github.com/Har3304/Langchain.git)
cd Langchain/Lead_Qualifier
