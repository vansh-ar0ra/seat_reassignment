# Migration Brief: Ollama → HuggingFace Inference

**Project:** IRROPS Flight Rebooking environment (OpenEnv hackathon)
**From:** Local Ollama running `qwen2.5:3b`
**To:** HuggingFace-routed `Qwen/Qwen3-4B-Instruct-2507` via OpenAI client

## Why

The OpenEnv hackathon spec mandates the OpenAI client pointed at the HF router using `MODEL_NAME` and `HF_TOKEN`. The submission's `inference.py` must also run on a 2 vCPU / 8 GB box in under 20 min, so local model serving isn't viable. Hosted inference satisfies both.

## Pre-flight: confirm 3B is on the router

The HF router curates which models are served — sub-7B variants aren't always available because providers prioritize commercially relevant sizes. Verify before committing:

```bash
curl -s https://router.huggingface.co/v1/models \
  -H "Authorization: Bearer $HF_TOKEN" \
  | grep -i "qwen2.5-3b"
```

If nothing returns, switch `MODEL_NAME` to `Qwen/Qwen3-4B-Instruct-2507` — that's the smallest reliably-routed Qwen2.5 size and behaves similarly enough for our XML-CoT scaffolding.

## Setup

### 1. HF token

HuggingFace Settings → Access Tokens → create one with the **"Make calls to Inference Providers"** scope. Copy the `hf_...` value.

### 2. Create `.env` at repo root

```ini
# .env  (DO NOT COMMIT)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Add to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

Commit a sanitized template so Shreya and Anushka can clone and fill in:

```ini
# .env.example  (commit this one)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
HF_TOKEN=
```

### 3. Install dependencies

```bash
uv add openai python-dotenv
# or
pip install openai python-dotenv
```

### 4. Load env vars in `inference.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent, so submission infra still works

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]
```

`load_dotenv()` only fills in vars that aren't already set, so the same script runs unchanged when the hackathon runner injects env vars directly.

## Client code

Drop-in replacement for the existing Ollama `requests.post`:

```python
from openai import OpenAI

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def llm_chat(messages, temperature=0.7, max_tokens=1024, seed=42):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    return (resp.choices[0].message.content or "").strip() or None
```

**Parameter mapping from old Ollama call:**

| Ollama | OpenAI client |
|---|---|
| `options.num_predict` | `max_tokens` |
| `options.temperature` | `temperature` |
| `options.seed` | `seed` |
| `stream: false` | default; omit |
| `think: false` | not needed (Qwen2.5 has no thinking mode) |

No `<think>` regex strip needed — that was a Qwen3 quirk.

## Smoke test

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI; import os
c = OpenAI(base_url=os.environ['API_BASE_URL'], api_key=os.environ['HF_TOKEN'])
r = c.chat.completions.create(model=os.environ['MODEL_NAME'],
                              messages=[{'role':'user','content':'reply with the word pong'}],
                              max_tokens=10)
print(r.choices[0].message.content)
"
```

Expected: `pong` (or close to it). If you get HTTP 404 on the model, 3B isn't routed → bump to 7B-Instruct.

## Caveats

- **Determinism:** `seed` is accepted but most routed providers don't honour it strictly. Fix seeds in data generation, not the LLM call.
- **Rate limits:** quota is per-token-budget on your HF account. Run rollouts sequentially until you've checked headroom.
- **3B quality risk:** XML CoT scaffolding (`<observations>`, `<passenger_analysis>`, `<plan>`, etc.) sometimes drifts on a 3B — watch for unclosed tags and consider a tolerant XML parser before grading.