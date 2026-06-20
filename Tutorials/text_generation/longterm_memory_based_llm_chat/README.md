# Long-Term Memory Chatbot with LangChain & Llama-3.1

This repository contains a Python implementation of a conversational AI chatbot that retains its memory across distinct runs and sessions using an underlying SQLite database. 

It leverages **LangChain's Expression Language (LCEL)**, **Hugging Face Endpoints**, and **Meta's Llama-3.1-8B-Instruct** model.

## Features
* **Persistent Memory:** Messages are saved to a local SQLite database (`chat_history.db`) automatically. 
* **Session Isolation:** By specifying a unique `session_id`, multiple conversations can be maintained independently without overlapping context.
* **Open Source LLM Integration:** Powered by Hugging Face's serverless endpoint architecture.

---

## Architecture Diagram

The workflow layout utilizes LangChain's `RunnableWithMessageHistory` to intercept inputs, pull historical context, build a cohesive prompt, and store the updated conversation state back to SQLite:
