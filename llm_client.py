"""
Unified LLM client for extraction pipelines.

Provider is selected via env var LLM_PROVIDER:
  LLM_PROVIDER=gemini    -> Gemini API (default)
  LLM_PROVIDER=openai    -> OpenAI API
  LLM_PROVIDER=anthropic -> Anthropic API
  LLM_PROVIDER=groq      -> Groq API (OpenAI-compatible)
  LLM_PROVIDER=cerebras  -> Cerebras API (OpenAI-compatible, free 1M tok/day)

Default configs:
  Gemini:    gemini-3-flash-preview, t=0, max_tokens=1024
  OpenAI:    gpt-4o-mini, t=0, max_tokens=1024
  Anthropic: claude-haiku-4-5-20251001, t=0, max_tokens=1024
  Groq:      llama-3.3-70b-versatile, t=0, max_tokens=1024

Override via env: LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS.
"""

import asyncio
import os

import httpx


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.removesuffix("```").strip()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _gemini_thinking_config(model: str) -> dict:
    if "3-flash" in model or "3-pro" in model or "3.1" in model:
        return {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}
    if "2.5" in model:
        return {"thinkingConfig": {"thinkingBudget": 0}}
    return {}


def _parse_gemini_response(result: dict) -> str:
    parts = result["candidates"][0]["content"]["parts"]
    text = next(
        (p["text"] for p in reversed(parts) if not p.get("thought") and p.get("text")),
        parts[-1].get("text", ""),
    )
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.removesuffix("```").strip()


async def _gemini_call(prompt: str, *, timeout: float) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    model = os.environ.get("LLM_MODEL", "gemini-3-flash-preview")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    gen_cfg = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    gen_cfg.update(_gemini_thinking_config(model))

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Gemini API error {response.status_code}: {response.text[:200]}")
        result = response.json()

    try:
        return _parse_gemini_response(result)
    except (KeyError, IndexError, TypeError) as e:
        raise Exception(f"Gemini response format unexpected: {e} | body={str(result)[:200]}")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

async def _openai_call(prompt: str, *, timeout: float) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI API error {response.status_code}: {response.text[:200]}")
        result = response.json()

    try:
        return _strip_code_fences(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise Exception(f"OpenAI response format unexpected: {e} | body={str(result)[:200]}")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

async def _anthropic_call(prompt: str, *, timeout: float) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    model = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Anthropic API error {response.status_code}: {response.text[:200]}")
        result = response.json()

    try:
        return _strip_code_fences(result["content"][0]["text"])
    except (KeyError, IndexError, TypeError) as e:
        raise Exception(f"Anthropic response format unexpected: {e} | body={str(result)[:200]}")


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible API)
# ---------------------------------------------------------------------------

async def _groq_call(prompt: str, *, timeout: float) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    # gpt-oss / GLM are reasoning models: they spend tokens on chain-of-thought
    # before the answer. With a small budget the answer comes back empty. Give
    # headroom and keep reasoning effort low so the JSON answer always fits.
    _reasoning = "gpt-oss" in model or "glm" in model.lower()
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "3000" if _reasoning else "1024"))

    base_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _reasoning:
        base_payload["reasoning_effort"] = "low"
    # Groq rejects json_object response_format unless the prompt mentions JSON.
    # Only request it for prompts that actually ask for JSON (e.g. TBG extraction).
    want_json = "json" in prompt.lower()

    # Robust loop so NO turn is silently lost (clean cross-flag deltas require it):
    #  - 429 (TPM cap) / 5xx  -> backoff + retry
    #  - 400 json_validate_failed -> Groq's strict json-mode failed; drop
    #    response_format=json_object and retry (caller parses with _clean_json),
    #    then plain backoff retries. Prompt still asks for JSON, so output stays JSON-ish.
    last_err = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(5):
            payload = dict(base_payload)
            if want_json:
                payload["response_format"] = {"type": "json_object"}
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            sc = response.status_code
            if sc == 429 or sc >= 500:
                last_err = f"{sc}: {response.text[:160]}"
                await asyncio.sleep(6 * (attempt + 1))
                continue
            if sc == 400 and "json_validate_failed" in response.text:
                last_err = f"400 json_validate_failed: {response.text[:120]}"
                if want_json:
                    want_json = False          # drop strict json-mode, retry immediately
                    continue
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if sc != 200:
                raise Exception(f"Groq API error {sc}: {response.text[:200]}")
            result = response.json()
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise Exception(f"Groq response format unexpected: {e} | body={str(result)[:200]}")
            if not content:
                # Reasoning model spent the whole budget on CoT — bump budget and retry.
                last_err = (f"empty content "
                            f"(finish={result['choices'][0].get('finish_reason')})")
                base_payload["max_tokens"] = min(8000, int(base_payload["max_tokens"] * 1.6))
                await asyncio.sleep(2 * (attempt + 1))
                continue
            return _strip_code_fences(content)
    raise Exception(f"Groq API error after retries: {last_err}")


# ---------------------------------------------------------------------------
# Cerebras (OpenAI-compatible API) — free tier: 1M tokens/day, 30 RPM, no card.
# Free-tier context cap is 8K (fine for TBG's short prompts).
# ---------------------------------------------------------------------------

async def _cerebras_call(prompt: str, *, timeout: float) -> str:
    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set")

    model = os.environ.get("LLM_MODEL", "zai-glm-4.7")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    # Both Cerebras free models (gpt-oss, GLM) are reasoning models that spend
    # budget on chain-of-thought before the answer; give headroom so the JSON
    # content isn't truncated (kept under the 8K free-tier context cap).
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "5000"))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Cerebras supports OpenAI-style json_object; request it only for JSON prompts.
    if "json" in prompt.lower():
        payload["response_format"] = {"type": "json_object"}
    # Both free Cerebras models reason before answering — keep effort low so the
    # JSON answer fits the token budget instead of being truncated mid-thought.
    if "gpt-oss" in model or "glm" in model.lower():
        payload["reasoning_effort"] = "low"

    last_err = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(4):
            response = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            # Free tier RPM is tight — back off and retry on 429 / 5xx.
            if response.status_code == 429 or response.status_code >= 500:
                last_err = f"{response.status_code}: {response.text[:160]}"
                await asyncio.sleep(6 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise Exception(f"Cerebras API error {response.status_code}: {response.text[:200]}")
            result = response.json()
            try:
                msg = result["choices"][0]["message"]
                text = msg.get("content")
                if not text:
                    raise Exception(
                        f"Cerebras returned no content "
                        f"(finish={result['choices'][0].get('finish_reason')}); "
                        f"reasoning model may need higher LLM_MAX_TOKENS"
                    )
                return _strip_code_fences(text)
            except (KeyError, IndexError, TypeError) as e:
                raise Exception(f"Cerebras response format unexpected: {e} | body={str(result)[:200]}")
    raise Exception(f"Cerebras API error after retries: {last_err}")


# ---------------------------------------------------------------------------
# Public API: unified entry point
# ---------------------------------------------------------------------------

async def gemini_call(prompt: str, *, timeout: float = 30.0) -> str:
    """
    Unified LLM call. Provider selected via LLM_PROVIDER env var.
    Name kept as 'gemini_call' for backward compatibility with existing imports.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        return await _openai_call(prompt, timeout=timeout)
    if provider == "anthropic":
        return await _anthropic_call(prompt, timeout=timeout)
    if provider == "groq":
        return await _groq_call(prompt, timeout=timeout)
    if provider == "cerebras":
        return await _cerebras_call(prompt, timeout=timeout)
    return await _gemini_call(prompt, timeout=timeout)
