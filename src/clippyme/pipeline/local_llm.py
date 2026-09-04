"""Local LLM provider — viral detection with no API key and no per-job cost.

Gemini is the only part of a ClippyMe job that costs money; everything else
(transcription via Faster-Whisper, reframe, render, Smart Cut, upscale) already
runs locally for free. This module closes that last gap by pointing the same
prompt at a model running on your own hardware.

It speaks the OpenAI-compatible ``/chat/completions`` shape, which is what
Ollama (``http://localhost:11434/v1``), LM Studio, llama.cpp's server and vLLM
all expose — one implementation covers every common local runner. Transport is
stdlib ``urllib`` on purpose: a free-by-design path should not drag in a new
pinned dependency.

Nothing about the *prompt* changes. ``gemini_request.build_viral_prompt`` still
builds it and ``gemini_parser`` still parses the answer through its five-level
repair chain — this only swaps who answers. ``LocalResponse`` therefore
deliberately mirrors the google-genai response shape (``.text``,
``.usage_metadata.prompt_token_count`` / ``.candidates_token_count``) so the
caller's cost and parsing code needs no branching.

The pure halves (payload building, response parsing, config resolution) carry
no imports beyond the stdlib and are host-tested; only ``generate`` touches the
network.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:14b-instruct"
DEFAULT_TIMEOUT_SECONDS = 600
# The viral-detection prompt asks for a long structured JSON answer over a
# whole transcript; a small cap truncates it mid-object and guarantees a parse
# failure, so the default is generous.
DEFAULT_MAX_TOKENS = 8192
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def provider() -> str:
    """Which LLM answers the viral-detection prompt: ``gemini`` or ``local``.

    Defaults to ``gemini`` so existing deployments are untouched by an upgrade;
    set ``LLM_PROVIDER=local`` for a zero-cost job.
    """
    raw = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    return "local" if raw in ("local", "ollama", "lmstudio", "llamacpp", "vllm") else "gemini"


def is_local() -> bool:
    return provider() == "local"


def base_url() -> str:
    """Root of the OpenAI-compatible API, without a trailing slash.

    Operator-configured (env/compose), never user input, so it is expected to
    point at a private address — that is the whole point of running locally and
    is not the SSRF surface the download allow-list guards against.
    """
    raw = (os.getenv("LOCAL_LLM_BASE_URL") or "").strip() or DEFAULT_BASE_URL
    return raw.rstrip("/")


def model_name() -> str:
    return (os.getenv("LOCAL_LLM_MODEL") or "").strip() or DEFAULT_MODEL


def timeout_seconds() -> int:
    raw = (os.getenv("LOCAL_LLM_TIMEOUT") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_SECONDS


def max_tokens() -> int:
    raw = (os.getenv("LOCAL_LLM_MAX_TOKENS") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class LocalUsage:
    """Mirrors google-genai's usage metadata field names on purpose."""

    prompt_token_count: int = 0
    candidates_token_count: int = 0


@dataclass(frozen=True)
class LocalResponse:
    """Duck-typed stand-in for a google-genai response."""

    text: str
    usage_metadata: LocalUsage = field(default_factory=LocalUsage)


def build_chat_payload(prompt: str, model: str | None = None) -> dict:
    """The request body. Pure — no I/O.

    ``temperature`` is low because this is an extraction task with a strict
    output contract, not a creative one: the prompt already carries the
    copywriting instructions, and sampling noise here mostly produces malformed
    JSON. ``stream`` is false so the answer arrives as one parseable object.
    """
    return {
        "model": model or model_name(),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": max_tokens(),
    }


def parse_chat_response(data: dict) -> LocalResponse:
    """Turn an OpenAI-compatible chat completion into a ``LocalResponse``.

    Pure. Missing usage counters degrade to zero rather than raising: a local
    server that omits them is still a perfectly good answer, and the cost is
    zero either way.
    """
    if not isinstance(data, dict):
        raise ValueError("local LLM response was not a JSON object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        # Surface the server's own error text — "model not found" is the single
        # most common setup mistake and the message says exactly that.
        detail = data.get("error") or data.get("message") or data
        raise ValueError(f"local LLM returned no choices: {detail}")

    message = (choices[0] or {}).get("message") or {}
    text = message.get("content")
    if not isinstance(text, str):
        raise ValueError("local LLM response had no message content")

    usage = data.get("usage") or {}
    return LocalResponse(
        text=text,
        usage_metadata=LocalUsage(
            prompt_token_count=int(usage.get("prompt_tokens") or 0),
            candidates_token_count=int(usage.get("completion_tokens") or 0),
        ),
    )


def local_cost_analysis(response: LocalResponse, model: str | None = None) -> dict:
    """Cost record for a local generation: token counts, and zero spend.

    Shaped like ``gemini_request.compute_gemini_cost`` so the metadata file and
    the dashboard read it without a special case — the numbers are just zero.
    """
    usage = response.usage_metadata
    return {
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model or model_name(),
        "note": "Local model — no API cost",
    }


def generate(prompt: str, *, model: str | None = None, log_fn=print) -> LocalResponse:
    """POST the prompt to the local server and return the parsed answer.

    Raises ``RuntimeError`` with an actionable message when the server can't be
    reached — an unreachable local model is a setup problem the operator can
    fix, and saying so beats a bare connection error in the job log.
    """
    url = f"{base_url()}/chat/completions"
    body = json.dumps(build_chat_payload(prompt, model)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # Most local servers ignore auth entirely; some (vLLM behind a gateway)
    # want a bearer token, so pass one through when it is configured.
    api_key = (os.getenv("LOCAL_LLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    log_fn(f"🖥️  Local LLM: {model or model_name()} at {base_url()}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds()) as response:  # nosec B310: operator-configured local endpoint
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(8192).decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"local LLM at {base_url()} returned HTTP {exc.code}. {detail[:400]}"
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"could not reach the local LLM at {base_url()} ({exc}). "
            "Start it (e.g. `ollama serve`) or set LOCAL_LLM_BASE_URL to where it runs."
        ) from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("local LLM response exceeded the size cap")

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local LLM returned a non-JSON body: {exc}") from exc

    return parse_chat_response(data)
