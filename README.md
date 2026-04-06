# OpenEnv Seat Swap Inference

This project implements an OpenEnv-compatible server for a "Seat Swap" logic test, as well as an inference client script that solves it using Google's Gemini models.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (for environment management)
- Python 3.10+
- A Google Gemini API Key

## Environment Setup

Ensure you have your environment variables set correctly. You can create a `.env` file or export them directly in your shell.

```bash
# Required for inference using Gemini
export GEMINI_API_KEY="your_api_key_here"

# (Optional) If you have a wandb or similar integration
# export WANDB_API_KEY="your_api_key_here"
```

## Running the Components

All OpenEnv implementations require running the test environment server locally, and then running the agent/inference client alongside it.

### 1. Start the Environment Server

Use `uv` to start the server. This will automatically install dependencies based on `pyproject.toml` and start the server at `http://localhost:8000`.

```bash
uv run server
```

*Leave this running in a termainal tab!*

### 2. Run Gemini Inference 

In a new terminal window, with the server running, launch the Gemini inference agent.

```bash
uv run python inference_gemini_debug.py
```

This will run the most up-to-date version of the inference loop with debug logging, giving you insight into observations, LLM responses, and prompt management!
