# LLM Observability & Monitoring with LangChain and Hugging Face

This repository demonstrates how to implement a production-ready **Structured Logging** and **Telemetry Metrics Collection** pipeline for Large Language Models (LLMs) using LangChain, ChatHuggingFace, and Python's native logging framework.

The project wraps an LLM endpoint (`meta-llama/Llama-3.1-8B`) inside an instrumented class that automatically captures query latency, calculated token usage, errors, and cache hit metrics, emitting structured JSON logs for backend consumption.

---

## Architecture Overview

The framework consists of three decoupled components acting together to provide full observability over LLM execution:

[User Request] ──> [InstrumentedLLM] ──> (Calculates Latency & Tokens)
│
├──> [JSONFormatter] ────> Stream Output (JSON Logs)
│
└──> [MetricsCollector] ─> State Update (Aggregated Metrics)

1. **`JSONFormatter`**: Overrides standard textual logs to output machine-readable JSON format, seamlessly including contextual fields (`timestamp`, `level`, `module`, `function`) and dynamic custom parameters (`latency_ms`, `input_tokens`, `output_tokens`).
2. **`MetricsCollector`**: Aggregates running application-level statistics including total requests, execution error rates, average latency counters, total token throughput, and cache efficiency rates.
3. **`InstrumentedLLM`**: A defensive proxy wrapper around `ChatHuggingFace`. It intercepts downstream execution invocations, manages systemic run-time exceptions gracefully, and records execution metrics.

---

## Installation & Setup

### 1. Clone or Download the Files
Ensure your project workspace contains the application script along with the following installation manifests:
* `app.py` (Contains the main implementation and monitoring demo)
* `requirements.txt` (Third-party library dependencies)
* `README.md` (This documentation file)

### 2. Install Dependencies
Install the required upstream libraries using `pip`:
```bash
pip install -r requirements.txt
