"""Host tests for Gemini auth resolution (API key vs Vertex/OAuth).

``resolve_auth`` is pure and does not import google-genai, so the decision
logic is testable without the SDK or any credentials.
"""
import pytest

from clippyme.pipeline import gemini_auth


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GEMINI_AUTH_MODE", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT",
                "GCLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


# --- mode selection -----------------------------------------------------------

def test_defaults_to_api_key():
    """Existing deployments must not be moved onto a different auth path by an
    upgrade."""
    assert gemini_auth.auth_mode() == "api_key"
    assert gemini_auth.is_vertex() is False


@pytest.mark.parametrize("value", ["vertex", "vertexai", "vertex_ai", "vertex-ai",
                                   "oauth", "adc", "gcloud", "  OAuth  "])
def test_keyless_aliases_all_select_vertex(monkeypatch, value):
    """'oauth' and 'adc' are what people reach for; they mean the same Vertex +
    Application Default Credentials route, so accept the synonyms."""
    monkeypatch.setenv("GEMINI_AUTH_MODE", value)
    assert gemini_auth.is_vertex() is True


@pytest.mark.parametrize("value", ["api_key", "", "apikey", "nonsense"])
def test_anything_else_stays_on_the_api_key_path(monkeypatch, value):
    monkeypatch.setenv("GEMINI_AUTH_MODE", value)
    assert gemini_auth.auth_mode() == "api_key"


# --- api_key mode -------------------------------------------------------------

def test_explicit_key_beats_the_environment(monkeypatch):
    """The per-request X-Gemini-Key header is the caller's choice for that job."""
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    auth = gemini_auth.resolve_auth("from-header")
    assert auth.ok and auth.api_key == "from-header"
    assert auth.client_kwargs() == {"api_key": "from-header"}


def test_falls_back_to_the_environment_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert gemini_auth.resolve_auth().api_key == "from-env"


def test_missing_key_reports_both_ways_out():
    auth = gemini_auth.resolve_auth()
    assert not auth.ok
    assert "Settings" in auth.error and "GEMINI_AUTH_MODE=vertex" in auth.error


# --- vertex / OAuth mode ------------------------------------------------------

def test_vertex_needs_no_api_key(monkeypatch):
    """The whole point: credentials come from ADC, so an absent key is fine."""
    monkeypatch.setenv("GEMINI_AUTH_MODE", "oauth")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    auth = gemini_auth.resolve_auth()
    assert auth.ok
    assert auth.client_kwargs() == {"vertexai": True, "project": "my-proj", "location": "us-central1"}


def test_vertex_without_a_project_says_how_to_sign_in(monkeypatch):
    monkeypatch.setenv("GEMINI_AUTH_MODE", "vertex")
    auth = gemini_auth.resolve_auth()
    assert not auth.ok
    assert "GOOGLE_CLOUD_PROJECT" in auth.error
    assert "application-default login" in auth.error


def test_vertex_location_override_and_legacy_project_var(monkeypatch):
    monkeypatch.setenv("GEMINI_AUTH_MODE", "vertex")
    monkeypatch.setenv("GCLOUD_PROJECT", "legacy-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    auth = gemini_auth.resolve_auth()
    assert auth.project == "legacy-proj" and auth.location == "europe-west4"


def test_vertex_ignores_a_stray_api_key(monkeypatch):
    """A key left in Settings from an earlier setup must not silently put the
    job back on the metered path."""
    monkeypatch.setenv("GEMINI_AUTH_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    auth = gemini_auth.resolve_auth("AIza-leftover-key")
    assert auth.mode == "vertex"
    assert "api_key" not in auth.client_kwargs()


# --- describe() ---------------------------------------------------------------

def test_describe_never_leaks_the_key(monkeypatch):
    """It goes into the job log, which users paste into issues."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSUPERSECRETVALUE1234567890")
    assert "AIza" not in gemini_auth.resolve_auth().describe()


def test_describe_names_the_project_in_vertex_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_AUTH_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    assert "my-proj" in gemini_auth.resolve_auth().describe()


# --- requires_api_key: the API-layer gate -------------------------------------

def test_api_key_required_on_the_default_path():
    assert gemini_auth.requires_api_key() is True


def test_no_key_required_in_vertex_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_AUTH_MODE", "vertex")
    assert gemini_auth.requires_api_key() is False


def test_no_key_required_when_the_llm_is_local(monkeypatch):
    """LLM_PROVIDER=local doesn't call Gemini at all — demanding a Gemini key
    to submit the job would lock the free path out of the API entirely."""
    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert gemini_auth.requires_api_key() is False


def test_build_client_refuses_an_unusable_config():
    """The resolution's message is written for the operator, so build_client
    raises with it rather than letting the SDK fail obscurely later."""
    with pytest.raises(ValueError, match="No Gemini API key"):
        gemini_auth.build_client()
