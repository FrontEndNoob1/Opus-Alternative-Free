"""Host tests for the pure half of the 4K upscale path.

``plan_scale`` and ``estimate_peak_bytes`` do no I/O, so they run in the fast
suite; the ffmpeg/Real-ESRGAN orchestration around them needs real binaries
and is exercised by hand / in the Docker integration environment.
"""
import pytest

from clippyme.domain import upscale
from clippyme.domain.errors import ConflictError


# --- plan_scale ---------------------------------------------------------------

def test_plan_scale_defaults_to_2x_for_the_pipeline_native_render():
    """1080x1920 is what every ClippyMe job renders, and 2x is exactly the
    4K vertical (2160x3840) the feature promises."""
    assert upscale.plan_scale(1080, 1920) == 2


def test_plan_scale_picks_the_smallest_factor_that_reaches_4k():
    # 1280 long edge needs 3x (2x would stop at 2560, short of 3840).
    assert upscale.plan_scale(720, 1280) == 3


def test_plan_scale_caps_a_request_that_would_overshoot_4k():
    """4x on a 1080x1920 clip is an 8K render — 4x the pixels, time and disk
    of the 4K the endpoint advertises. The request is capped, not honoured."""
    assert upscale.plan_scale(1080, 1920, 4) == 2


def test_plan_scale_honours_a_request_that_does_not_overshoot():
    assert upscale.plan_scale(720, 1280, 2) == 2


def test_plan_scale_rejects_a_clip_already_at_4k():
    with pytest.raises(ValueError, match="already"):
        upscale.plan_scale(2160, 3840)


def test_plan_scale_rejects_an_out_of_range_factor():
    with pytest.raises(ValueError, match="scale must be"):
        upscale.plan_scale(1080, 1920, 5)


def test_upscale_clip_to_4k_maps_an_already_4k_clip_to_409(monkeypatch):
    """The ValueError has to surface as a ConflictError so the API returns 409
    rather than a 500 — a second click on an upscaled clip is a normal state,
    not a server fault."""
    monkeypatch.setattr(upscale, "probe_dimensions", lambda _p: (2160, 3840))
    with pytest.raises(ConflictError):
        upscale.upscale_clip_to_4k("clip.mp4", "out.mp4")


# --- estimate_peak_bytes ------------------------------------------------------

def test_estimate_peak_counts_both_frame_sets():
    """Source and upscaled frames coexist on disk, and the upscaled set is
    scale^2 larger — the estimate must reflect both, not just one."""
    one = upscale.estimate_peak_bytes(1080, 1920, 1, 2)
    per_source = 1080 * 1920 * 3 * upscale._PNG_RAW_RATIO
    assert one == pytest.approx(per_source * (1 + 4), rel=1e-6)


def test_estimate_peak_scales_with_frame_count():
    single = upscale.estimate_peak_bytes(1080, 1920, 1, 2)
    assert upscale.estimate_peak_bytes(1080, 1920, 100, 2) == 100 * single


def test_estimate_peak_for_a_full_length_clip_is_gigabytes():
    """Guards the assumption behind the pre-spend disk check: a 60s vertical
    clip stages tens of GB, which is why it can't quietly live in /tmp."""
    peak_gb = upscale.estimate_peak_bytes(1080, 1920, 60 * 30, 2) / 1e9
    assert peak_gb > 10


# --- env knobs ----------------------------------------------------------------

def test_upscale_timeout_defaults_to_the_shipped_nginx_read_timeout(monkeypatch):
    """600s is dashboard/nginx.conf's proxy_read_timeout — a larger default
    would let the render outlive the 504 the browser already got."""
    monkeypatch.delenv("CLIPPYME_UPSCALE_TIMEOUT_SECONDS", raising=False)
    assert upscale.upscale_timeout() == 600


@pytest.mark.parametrize("raw", ["", "0", "-5", "abc"])
def test_upscale_timeout_ignores_junk_values(monkeypatch, raw):
    monkeypatch.setenv("CLIPPYME_UPSCALE_TIMEOUT_SECONDS", raw)
    assert upscale.upscale_timeout() == 600


def test_upscale_timeout_honours_a_valid_override(monkeypatch):
    monkeypatch.setenv("CLIPPYME_UPSCALE_TIMEOUT_SECONDS", "1800")
    assert upscale.upscale_timeout() == 1800


def test_max_clip_duration_default_and_override(monkeypatch):
    monkeypatch.delenv("CLIPPYME_UPSCALE_MAX_DURATION_SECONDS", raising=False)
    assert upscale.max_clip_duration_seconds() == 90
    monkeypatch.setenv("CLIPPYME_UPSCALE_MAX_DURATION_SECONDS", "45")
    assert upscale.max_clip_duration_seconds() == 45


def test_default_model_is_the_general_purpose_one(monkeypatch):
    """The anime-tuned models mangle real faces — the live-action default
    matters for talking-head/stream clips."""
    monkeypatch.delenv("CLIPPYME_UPSCALE_MODEL", raising=False)
    assert upscale.default_model() == "realesrgan-x4plus"
