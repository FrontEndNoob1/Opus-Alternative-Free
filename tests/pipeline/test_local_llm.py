"""Host tests for the free local-LLM provider (no network, no key).

Everything here is the pure half — config resolution, payload building,
response parsing, the zero-cost record. ``generate`` itself shells out over
HTTP and is exercised against a real local server by hand.
"""
import pytest

from clippyme.pipeline import local_llm


# --- provider selection -------------------------------------------------------

def test_provider_defaults_to_gemini(monkeypatch):
    """An upgrade must not silently move existing deployments onto a local
    model they haven't configured."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert local_llm.provider() == "gemini"
    assert local_llm.is_local() is False


@pytest.mark.parametrize("value", ["local", "ollama", "LMStudio", "  vllm  ", "llamacpp"])
def test_provider_recognises_the_local_runners(monkeypatch, value):
    monkeypatch.setenv("LLM_PROVIDER", value)
    assert local_llm.is_local() is True


@pytest.mark.parametrize("value", ["gemini", "", "openai", "nonsense"])
def test_provider_falls_back_to_gemini_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("LLM_PROVIDER", value)
    assert local_llm.provider() == "gemini"


# --- config -------------------------------------------------------------------

def test_base_url_defaults_to_ollama_and_strips_trailing_slash(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    assert local_llm.base_url() == "http://localhost:11434/v1"
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://box.lan:1234/v1/")
    assert local_llm.base_url() == "http://box.lan:1234/v1"


@pytest.mark.parametrize("raw", ["", "0", "-30", "abc"])
def test_timeout_ignores_junk(monkeypatch, raw):
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT", raw)
    assert local_llm.timeout_seconds() == 600


def test_timeout_and_max_tokens_honour_overrides(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT", "1200")
    monkeypatch.setenv("LOCAL_LLM_MAX_TOKENS", "16384")
    assert local_llm.timeout_seconds() == 1200
    assert local_llm.max_tokens() == 16384


# --- payload ------------------------------------------------------------------

def test_payload_is_non_streaming(monkeypatch):
    """The parser needs one complete JSON object; a streamed answer would
    arrive as chunks it can't parse."""
    monkeypatch.delenv("LOCAL_LLM_MAX_TOKENS", raising=False)
    payload = local_llm.build_chat_payload("PROMPT", model="m")
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert payload["model"] == "m"


def test_payload_keeps_temperature_low():
    """Extraction against a strict output contract — sampling noise here shows
    up as malformed JSON, not creativity."""
    assert local_llm.build_chat_payload("p")["temperature"] <= 0.4


# --- response parsing ---------------------------------------------------------

def _completion(content, **usage):
    return {"choices": [{"message": {"content": content}}], "usage": usage}


def test_parse_uses_the_genai_usage_field_names():
    """The caller's cost block reads .usage_metadata.prompt_token_count — the
    whole point of the shim is that it needs no branching."""
    parsed = local_llm.parse_chat_response(
        _completion("hello", prompt_tokens=101, completion_tokens=7))
    assert parsed.text == "hello"
    assert parsed.usage_metadata.prompt_token_count == 101
    assert parsed.usage_metadata.candidates_token_count == 7


def test_parse_tolerates_a_server_that_omits_usage():
    """Some local runners don't report token counts; the answer is still good
    and the cost is zero either way."""
    parsed = local_llm.parse_chat_response(_completion("hi"))
    assert parsed.usage_metadata.prompt_token_count == 0


def test_parse_surfaces_the_servers_own_error():
    """'model not found' is the most common setup mistake — the message has to
    reach the job log instead of a generic parse failure."""
    with pytest.raises(ValueError, match="no choices"):
        local_llm.parse_chat_response({"error": "model 'qwen' not found, try pulling it"})


@pytest.mark.parametrize("payload", [
    {"choices": []},
    {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": None}}]},
    "not a dict",
])
def test_parse_rejects_malformed_bodies(payload):
    with pytest.raises(ValueError):
        local_llm.parse_chat_response(payload)


# --- cost ---------------------------------------------------------------------

def test_local_cost_is_zero_but_keeps_the_token_counts():
    """Same record shape as compute_gemini_cost so metadata and the dashboard
    read it unbranched — the numbers are just zero."""
    parsed = local_llm.parse_chat_response(
        _completion("x", prompt_tokens=24817, completion_tokens=512))
    cost = local_llm.local_cost_analysis(parsed, model="qwen2.5:14b-instruct")
    assert cost["total_cost"] == 0.0
    assert cost["input_cost"] == 0.0 and cost["output_cost"] == 0.0
    assert cost["input_tokens"] == 24817
    assert cost["model"] == "qwen2.5:14b-instruct"
    assert "no API cost" in cost["note"]


def test_local_cost_record_matches_the_gemini_record_keys():
    from clippyme.pipeline.gemini_request import compute_gemini_cost

    gemini_keys = set(compute_gemini_cost(10, 5, "gemini-2.5-flash"))
    local_keys = set(local_llm.local_cost_analysis(local_llm.LocalResponse(text="x")))
    assert gemini_keys <= local_keys
