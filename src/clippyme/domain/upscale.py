"""4K AI upscale for a finished clip via Real-ESRGAN (free, local, no API cost).

Impure orchestrator (shells out to ffmpeg + the Real-ESRGAN binary) — not
host-tested, same category as ``smartcut.py``. Frame-based, because the
Real-ESRGAN ncnn-vulkan CLI has no direct video mode:

  1. ffmpeg extracts the clip to lossless PNG frames.
  2. ``realesrgan-ncnn-vulkan`` upscales each frame (Vulkan GPU when present,
     software Vulkan/llvmpipe on CPU-only hosts — much slower but still free).
  3. ffmpeg reassembles the upscaled frames at the shared x264 settings and
     re-muxes the ORIGINAL audio stream untouched.

The input here is always ClippyMe's own already-rendered clip (a fresh
constant-frame-rate libx264 encode per ``domain.encode`` — never the raw
downloaded source), so VFR compensation is a non-issue.

This is deliberately NOT part of the automatic per-job pipeline: per-frame AI
upscaling is far slower than any other render pass (minutes on GPU, potentially
much longer on CPU-only hosts), so it only ever runs on an explicit per-clip
request (``POST /api/upscale/{job_id}/{clip_index}``, see ``upscale_service``).
"""
import logging
import os
import subprocess
import tempfile

from clippyme.domain.encode import ffmpeg_timeout, x264_video_args
from clippyme.domain.errors import ConflictError, UpscaleError
from clippyme.integrations import realesrgan_provisioner
from clippyme.pipeline.media_probe import probe_dimensions

logger = logging.getLogger(__name__)

VALID_SCALES = (2, 3, 4)
# 3840px long edge is the standard "4K"/UHD threshold regardless of
# orientation — a vertical 2160x3840 clip is exactly 2x ClippyMe's default
# 1080x1920 render, which is why scale=2 is the default.
MAX_LONG_EDGE = 3840

_DEFAULT_MODEL = "realesrgan-x4plus"
_DEFAULT_TIMEOUT_SECONDS = 3600
_DEFAULT_MAX_DURATION_SECONDS = 90


def default_model() -> str:
    """Resolved Real-ESRGAN model — ``CLIPPYME_UPSCALE_MODEL`` or realesrgan-x4plus
    (the general-purpose photo/live-action model; good default for talking-head
    and stream clips). Real-ESRGAN also ships anime-tuned models
    (realesrgan-x4plus-anime, realesr-animevideov3) for animated content."""
    return (os.getenv("CLIPPYME_UPSCALE_MODEL") or "").strip() or _DEFAULT_MODEL


def upscale_timeout() -> int:
    """Per-clip AI-pass timeout in seconds — ``CLIPPYME_UPSCALE_TIMEOUT_SECONDS``
    (>0) or 3600. Deliberately far larger than ``ffmpeg_timeout()``: a per-frame
    neural net pass on a CPU-only host can legitimately take a long time."""
    raw = (os.getenv("CLIPPYME_UPSCALE_TIMEOUT_SECONDS") or "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_TIMEOUT_SECONDS


def max_clip_duration_seconds() -> int:
    """Longest clip duration eligible for upscaling —
    ``CLIPPYME_UPSCALE_MAX_DURATION_SECONDS`` (>0) or 90. Guards a shared host
    against a pathologically long per-frame AI job; ClippyMe clips are 15-60s
    by construction so this only ever blocks unusual/legacy clips."""
    raw = (os.getenv("CLIPPYME_UPSCALE_MAX_DURATION_SECONDS") or "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_MAX_DURATION_SECONDS


def plan_scale(width: int, height: int, requested_scale: int | None = None) -> int:
    """Pick (or validate) the upscale factor. Pure — no I/O.

    Raises ``ValueError`` when the clip is already at/above the 4K long edge,
    or when ``requested_scale`` isn't one of ``VALID_SCALES``. With no
    ``requested_scale``, picks the smallest scale in ``VALID_SCALES`` whose
    result reaches ``MAX_LONG_EDGE``.
    """
    long_edge = max(int(width), int(height))
    if long_edge >= MAX_LONG_EDGE:
        raise ValueError(f"Clip is already {width}x{height} — at/above 4K, nothing to upscale")
    if requested_scale is not None:
        if requested_scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {VALID_SCALES}")
        return requested_scale
    for scale in VALID_SCALES:
        if long_edge * scale >= MAX_LONG_EDGE:
            return scale
    return VALID_SCALES[-1]


def _probe_r_frame_rate(path: str, default: str = "30/1") -> str:
    """Raw ``r_frame_rate`` string (e.g. ``"30000/1001"``) — passed straight
    through to ffmpeg's ``-framerate`` so no float rounding drift is introduced.
    Degrades to ``default`` on any failure, same convention as media_probe's
    probe_* helpers."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=s=x:p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        value = result.stdout.strip().split("\n")[0] if result.returncode == 0 else ""
        return value or default
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return default


def _run(cmd: list[str], *, timeout: int, step: str) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise UpscaleError(f"Upscale {step} timed out", status_code=504)
    except OSError as e:
        raise UpscaleError(f"Upscale {step} failed to launch: {e}")
    if result.returncode != 0:
        # Full output stays server-side only (paths/tracebacks) — never in the
        # client-facing error, matching reframe_service's convention.
        logger.error(
            "Upscale %s failed (code %s):\n%s", step, result.returncode,
            (result.stderr or result.stdout or b"").decode(errors="replace")[-2000:],
        )
        raise UpscaleError(f"Upscale {step} failed. Check server logs for details.")


def upscale_video(input_path: str, output_path: str, *, scale: int,
                   model: str | None = None, binary_path: str | None = None) -> None:
    """Re-render ``input_path`` at ``scale``x resolution via Real-ESRGAN into
    ``output_path``. Raises ``UpscaleError`` on any failure. Caller owns
    picking ``scale`` (see ``plan_scale``) and atomically placing the result.
    """
    binary_path = binary_path or realesrgan_provisioner.ensure_binary()
    if not binary_path:
        raise UpscaleError(
            "Real-ESRGAN is not installed on this server yet and could not be "
            "auto-provisioned (see server logs) — 4K upscaling is unavailable.",
            status_code=503,
        )

    model = model or default_model()
    fps = _probe_r_frame_rate(input_path)

    with tempfile.TemporaryDirectory(prefix="clippyme-upscale-") as tmp:
        frames_in = os.path.join(tmp, "in")
        frames_out = os.path.join(tmp, "out")
        os.makedirs(frames_in, exist_ok=True)
        os.makedirs(frames_out, exist_ok=True)

        _run(
            ["ffmpeg", "-y", "-i", input_path, "-vsync", "0",
             os.path.join(frames_in, "f%08d.png")],
            timeout=ffmpeg_timeout(), step="frame extraction",
        )

        _run(
            [binary_path, "-i", frames_in, "-o", frames_out,
             "-n", model, "-s", str(scale), "-f", "png"],
            timeout=upscale_timeout(), step="AI upscale pass",
        )

        _run(
            ["ffmpeg", "-y",
             "-framerate", fps, "-i", os.path.join(frames_out, "f%08d.png"),
             "-i", input_path,
             "-map", "0:v:0", "-map", "1:a:0?",
             *x264_video_args(),
             "-c:a", "copy",
             "-r", fps,
             output_path],
            timeout=ffmpeg_timeout(), step="reassembly",
        )


def upscale_clip_to_4k(input_path: str, output_path: str, *, requested_scale: int | None = None) -> dict:
    """High-level entry point: probe → plan → render. Returns
    ``{"scale": int, "width": int, "height": int}`` for metadata persistence.
    """
    width, height = probe_dimensions(input_path)
    try:
        scale = plan_scale(width, height, requested_scale)
    except ValueError as e:
        raise ConflictError(str(e))

    upscale_video(input_path, output_path, scale=scale)
    return {"scale": scale, "width": width * scale, "height": height * scale}
