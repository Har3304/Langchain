# Conversational LLM with Memory (LangChain & Hugging Face)

This project demonstrates how to build a stateful conversational AI assistant using **LangChain** and a hosted **Hugging Face Endpoint**. By leveraging `RunnableWithMessageHistory`, the script remembers context across multiple turns in a single conversation session.

It uses the `meta-llama/Llama-3.1-8B-Instruct` model chat wrapper to ensure correct system prompt handling and conversation tracking.

## Features

* **Stateful Conversation:** Remembers previous user interactions via an in-memory chat message history (`InMemoryChatMessageHistory`).
* **Session-Based Tracking:** Keeps histories isolated using `session_id`, allowing multi-user configuration.
* **Hugging Face Integration:** Uses `ChatHuggingFace` wrapped around a remote `HuggingFaceEndpoint` for low-latency, API-driven text generation.

---

## Prerequisites

Before running the script, you need a **Hugging Face User Access Token** with permissions to access the Llama-3.1 model hierarchy.

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Set it as an environment variable in your terminal:

```bash
export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
