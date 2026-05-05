"""
Unified LLM client that talks to both local vLLM servers and remote APIs
via the OpenAI-compatible chat completions interface.

Supports both sync (single request) and async (batched) modes.
Async batching is critical for GPU utilization — sending concurrent
requests lets vLLM batch them on the GPU for 10-50x throughput.
"""
import asyncio
import os
import re
from openai import OpenAI, AsyncOpenAI


def make_client(model_cfg):
    """Create an OpenAI-compatible client from a model config dict."""
    if model_cfg["type"] == "local":
        base_url = f"http://localhost:{model_cfg['port']}/v1"
        return OpenAI(base_url=base_url, api_key="dummy")
    else:
        api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
        if not api_key:
            raise ValueError(f"Set env var {model_cfg['api_key_env']} for remote model")
        return OpenAI(base_url=model_cfg["base_url"], api_key=api_key)


def make_async_client(model_cfg):
    """Create an async OpenAI-compatible client for batched requests."""
    if model_cfg["type"] == "local":
        base_url = f"http://localhost:{model_cfg['port']}/v1"
        return AsyncOpenAI(base_url=base_url, api_key="dummy")
    else:
        api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
        if not api_key:
            raise ValueError(f"Set env var {model_cfg['api_key_env']} for remote model")
        return AsyncOpenAI(base_url=model_cfg["base_url"], api_key=api_key)


def get_model_id(model_cfg):
    """Return the model ID to pass in API calls."""
    if model_cfg["type"] == "local":
        return model_cfg["hf_id"]
    return model_cfg["model_id"]


# ============================================================
# Synchronous (single request) API
# ============================================================

def chat_single(client, model_id, prompt, temperature=1.0, max_tokens=10):
    """Single-turn chat: send prompt, return stripped response."""
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] API call failed: {e}")
        return None


def chat_multi(client, model_id, messages, prompt, temperature=1.0, max_tokens=10):
    """
    Multi-turn chat: append user prompt to messages, call API,
    append assistant response, return response text.
    Messages list is mutated in-place (same as paper's convention).
    """
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": content})
        return content
    except Exception as e:
        print(f"[Error] API call failed: {e}")
        messages.append({"role": "assistant", "content": ""})
        return None


# ============================================================
# Async (batched) API — for high GPU utilization
# ============================================================

async def _async_chat_single(async_client, model_id, prompt, temperature, max_tokens, semaphore):
    """Single async chat request with concurrency control."""
    async with semaphore:
        try:
            resp = await async_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return None


async def _batch_chat_single_async(model_cfg, model_id, prompts, temperature, max_tokens,
                                    max_concurrent=32):
    """Send many single-turn prompts concurrently. Returns list of responses.
    Creates a fresh AsyncOpenAI client inside the event loop to avoid
    httpx connection-pool issues when reusing clients across asyncio.run() calls.
    """
    # Create client inside the event loop so its httpx pool is tied to this loop
    if model_cfg["type"] == "local":
        base_url = f"http://localhost:{model_cfg['port']}/v1"
        api_key = "dummy"
    else:
        api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
        base_url = model_cfg["base_url"]

    async with AsyncOpenAI(base_url=base_url, api_key=api_key) as async_client:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            _async_chat_single(async_client, model_id, p, temperature, max_tokens, semaphore)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)


def batch_chat_single(model_cfg, model_id, prompts, temperature=1.0, max_tokens=10,
                      max_concurrent=32):
    """
    Send many single-turn prompts concurrently (sync wrapper).
    Returns list of response strings (same order as prompts).

    This is the main speedup: instead of 1 request at a time,
    we send up to max_concurrent requests simultaneously,
    letting vLLM batch them on the GPU.
    """
    return asyncio.run(
        _batch_chat_single_async(model_cfg, model_id, prompts, temperature,
                                  max_tokens, max_concurrent)
    )


# ============================================================
# Extraction
# ============================================================

def extract_yes_no(text):
    """Extract final yes/no answer from model output.

    Handles reasoning models that wrap output in <think>...</think> tags
    or produce long chain-of-thought before the answer.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip <think>...</think> blocks (Qwen3, DeepSeek-R1)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Also strip incomplete <think> blocks (truncated reasoning)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    # If nothing left after stripping thinking, try the raw text
    if not cleaned:
        cleaned = text.strip()

    # Remove trailing non-alpha chars
    tmp = cleaned
    while tmp and not tmp[-1].isalpha():
        tmp = tmp[:-1]
    if tmp:
        lower = tmp.lower()
        if lower.endswith("yes"):
            return "Yes"
        elif lower.endswith("no"):
            return "No"

    # Fallback: search for standalone Yes/No anywhere (last occurrence wins)
    matches = re.findall(r'\b(yes|no)\b', cleaned, re.IGNORECASE)
    if matches:
        last = matches[-1].lower()
        return "Yes" if last == "yes" else "No"

    return None
