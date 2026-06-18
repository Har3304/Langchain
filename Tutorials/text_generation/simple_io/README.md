# LangChain Chatbot with Llama-3.1-8b-Instruct on Google Colab

This repository contains a simple interactive chat loop built using **LangChain** and the **Hugging Face Endpoint API**. It utilizes the `meta-llama/Llama-3.1-8b-Instruct` model to create a helpful, witty AI assistant directly inside a Google Colab environment.

## Features
* **LCEL (LangChain Expression Language):** Uses modern pipe syntax (`prompt | bot`) for clean chain construction.
* **Hugging Face Integration:** Leverages `HuggingFaceEndpoint` and `ChatHuggingFace` for optimized chat-tuned responses.
* **Google Colab Native:** Safely manages API keys using Colab's built-in `userdata` secrets manager.

---

## Setup and Prerequisites

### 1. Hugging Face Access Token
Because `Llama-3.1-8b-Instruct` is a gated model, you need a Hugging Face account and an explicit access token to use it.
1. Go to your [Hugging Face Settings -> Tokens](https://huggingface.co/settings/tokens).
2. Generate a new token with **Read** access.
3. Visit the [Llama-3.1-8b-Instruct Model Page](https://huggingface.co/meta-llama/Llama-3.1-8b-Instruct) and accept the license agreement if you haven't already.

### 2. Configure Google Colab Secrets
To prevent leaking your API keys in code:
1. Open your Google Colab notebook.
2. Click on the **Key icon (Secrets)** in the left sidebar.
3. Add a new secret:
   * **Name:** `HF_TOKEN_READ`
   * **Value:** *Your Hugging Face Read Token*
4. Toggle on **Notebook access**.

---

## Installation

Before running the script, install the required packages inside your Colab notebook:

```bash
!pip install -r requirements.txt
