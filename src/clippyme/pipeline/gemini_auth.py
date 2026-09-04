"""How ClippyMe authenticates to Gemini: API key, or OAuth via Vertex AI.

Two modes, chosen with ``GEMINI_AUTH_MODE``:

``api_key`` (default)
    An AI Studio key (https://aistudio.google.com/apikey), pasted in Settings
    or set in the environment. Simplest path, unchanged behaviour.

``vertex``
    No key at all. The SDK talks to Vertex AI and authenticates with
    Application Default Credentials, which is the standard Google OAuth story:

      * ``gcloud auth application-default login`` — a real OAuth browser
        consent flow that writes user credentials to disk and refreshes them.
      * ``GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json`` — a service account.
      * The metadata server, when running on Google Cloud (nothing to set up).

    All three land in the same place, so ClippyMe does not implement an OAuth
    client of its own: hosting one would force every self-hoster to register a
    Google Cloud OAuth app, configure a consent screen and manage redirect
    URIs — strictly more setup than pasting a key, for the same result. Vertex
    mode needs only ``GOOGLE_CLOUD_PROJECT``.

``resolve_auth`` is pure and host-tested; only ``build_client`` imports the SDK,
so the decision logic can be tested without google-genai installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_LOCATION = "us-central1"

# Accepted spellings for the keyless path. "oauth" and "adc" are what users
# reach for; they all mean the same Vertex + Application Default Credentials
# route, so accept them rather than failing on a reasonable synonym.
_VERTEX_ALIASES = frozenset({"vertex", "vertexai", "vertex_ai", "oauth", "adc", "gcloud"})


def auth_mode() -> str:
    """``"vertex"`` or ``"api_key"``. Defaults to ``api_key``."""
    raw = (os.getenv("GEMINI_AUTH_MODE") or "").strip().lower().replace("-", "_")
    return "vertex" if raw in _VERTEX_ALIASES else "api_key"


def is_vertex() -> bool:
    return auth_mode() == "vertex"


def vertex_project() -> str:
    return (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT") or "").strip()


def vertex_location() -> str:
    return (os.getenv("GOOGLE_CLOUD_LOCATION") or "").strip() or DEFAULT_LOCATION


@dataclass(frozen=True)
class GeminiAuth:
    """Resolved auth. ``error`` is non-empty when the config can't be used."""

    mode: str
    api_key: str = ""
    project: str = ""
    location: str = DEFAULT_LOCATION
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def client_kwargs(self) -> dict:
        """Exactly what ``genai.Client(**kwargs)`` needs for this mode."""
        if self.mode == "vertex":
            return {"vertexai": True, "project": self.project, "location": self.location}
        return {"api_key": self.api_key}

    def describe(self) -> str:
        """One line for the job log. Never includes the key itself."""
        if self.mode == "vertex":
            return f"Vertex AI (OAuth/ADC) · project {self.project} · {self.location}"
        return "AI Studio API key"


def resolve_auth(api_key: str | None = None) -> GeminiAuth:
    """Work out how to authenticate. Pure — reads env, touches nothing else.

    ``api_key`` is the per-request key (the ``X-Gemini-Key`` header, or the
    value persisted in Settings); it wins over the environment in api_key mode.
    A resolution that can't work comes back with ``error`` set rather than
    raising, so callers can turn it into their own kind of failure.
    """
    if is_vertex():
        project = vertex_project()
        if not project:
            return GeminiAuth(
                mode="vertex",
                location=vertex_location(),
                error=(
                    "GEMINI_AUTH_MODE is set to vertex/OAuth but GOOGLE_CLOUD_PROJECT "
                    "is not set. Set it to your Google Cloud project id, and sign in "
                    "with `gcloud auth application-default login` (or point "
                    "GOOGLE_APPLICATION_CREDENTIALS at a service-account key)."
                ),
            )
        return GeminiAuth(mode="vertex", project=project, location=vertex_location())

    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return GeminiAuth(
            mode="api_key",
            error=(
                "No Gemini API key configured. Add one in Settings, or switch to "
                "keyless auth with GEMINI_AUTH_MODE=vertex + GOOGLE_CLOUD_PROJECT."
            ),
        )
    return GeminiAuth(mode="api_key", api_key=key)


def requires_api_key() -> bool:
    """Whether a job still needs the caller to supply a Gemini API key.

    False in Vertex/OAuth mode (credentials come from ADC) and false when
    ``LLM_PROVIDER=local`` (Gemini isn't involved at all). The API layer gates
    on this instead of demanding the header unconditionally — otherwise both
    keyless setups are locked out of submitting a job.
    """
    from clippyme.pipeline import local_llm

    if local_llm.is_local():
        return False
    return not is_vertex()


def build_client(api_key: str | None = None, *, http_options=None):
    """Construct a ``genai.Client`` for the resolved mode.

    Raises ``ValueError`` with the resolution's own message when the config is
    unusable — that message is written for the operator, so callers should
    surface it rather than replacing it.
    """
    auth = resolve_auth(api_key)
    if not auth.ok:
        raise ValueError(auth.error)

    from google import genai

    kwargs = auth.client_kwargs()
    if http_options is not None:
        kwargs["http_options"] = http_options
    return genai.Client(**kwargs)
