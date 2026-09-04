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
import shutil
import subprocess
import tempfile

from clippyme.domain.encode import ffmpeg_timeout, x264_video_args
from clippyme.domain.errors import ConflictError, UpscaleError
from clippyme.integrations import realesrgan_provisioner
from clippyme.pipeline.media_probe import parse_frame_rate, probe_dimensions, probe_duration

logger = logging.getLogger(__name__)

VALID_SCALES = (2, 3, 4)
# 3840px long edge is the standard "4K"/UHD threshold regardless of
# orientation — a vertical 2160x3840 clip is exactly 2x ClippyMe's default
# 1080x1920 render, which is why scale=2 is the default.
MAX_LONG_EDGE = 3840

_DEFAULT_MODEL = "realesrgan-x4plus"
# 600s matches the shipped nginx `proxy_read_timeout` (dashboard/nginx.conf).
# Going past it would let the render keep running server-side long after the
# proxy has already returned 504 to the browser — the user sees "upscale
# failed" while the work silently continues. Raise BOTH together, the same way
# MAX_FILE_SIZE_MB and nginx's client_max_body_size have to move in tandem.
_DEFAULT_TIMEOUT_SECONDS = 600
_DEFAULT_MAX_DURATION_SECONDS = 90


def default_model() -> str:
    """Resolved Real-ESRGAN model — ``CLIPPYME_UPSCALE_MODEL`` or realesrgan-x4plus
    (the general-purpose photo/live-action model; good default for talking-head
    and stream clips). Real-ESRGAN also ships anime-tuned models
    (realesrgan-x4plus-anime, realesr-animevideov3) for animated content."""
    return (os.getenv("CLIPPYME_UPSCALE_MODEL") or "").strip() or _DEFAULT_MODEL


def upscale_timeout() -> int:
    """Per-clip AI-pass timeout in seconds — ``CLIPPYME_UPSCALE_TIMEOUT_SECONDS``
    (>0) or 600, matching the shipped nginx read timeout (see the constant).
    A per-frame neural net pass on a CPU-only host can exceed this on longer
    clips; raise this AND nginx's ``proxy_read_timeout`` together if you need
    more, or run the upscale on a GPU host where it isn't close."""
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

    A ``requested_scale`` is CAPPED so the result never overshoots 4K: 4x on a
    1080x1920 clip would be a 4320x7680 (8K) render — 4x the pixels, render
    time and disk of the 4K the feature actually promises.
    """
    long_edge = max(int(width), int(height))
    if long_edge >= MAX_LONG_EDGE:
        raise ValueError(f"Clip is already {width}x{height} — at/above 4K, nothing to upscale")
    if requested_scale is not None and requested_scale not in VALID_SCALES:
        raise ValueError(f"scale must be one of {VALID_SCALES}")
    for scale in VALID_SCALES:
        if long_edge * scale >= MAX_LONG_EDGE:
            # Smallest scale that reaches 4K — also the cap for a request.
            return min(requested_scale, scale) if requested_scale else scale
    return requested_scale or VALID_SCALES[-1]


# Rough PNG size for one frame of real (noisy) video content, as a fraction of
# raw RGB. Measured against ClippyMe renders; synthetic/flat footage compresses
# far better, so this errs on the safe side for a pre-spend disk check.
_PNG_RAW_RATIO = 0.45


def estimate_peak_bytes(width: int, height: int, frame_count: int, scale: int) -> int:
    """Estimated peak disk for one upscale: the source PNG frames plus the
    ``scale``x upscaled ones, which coexist on disk. Pure — no I/O."""
    per_source_frame = width * height * 3 * _PNG_RAW_RATIO
    per_output_frame = per_source_frame * scale * scale
    return int(frame_count * (per_source_frame + per_output_frame))


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


def _probe_fps_float(path: str, default: float = 30.0) -> float:
    """Frame rate as a float, for frame-count estimation only. Degrades to
    ``default`` on any unreadable/zero value (never raises)."""
    rate = parse_frame_rate(_probe_r_frame_rate(path))
    return rate if rate and rate > 0 else default


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

    # Frames are staged NEXT TO the output clip, not in /tmp: a 60s 4K job
    # stages many GB of PNGs, and in the shipped container /tmp is the image's
    # writable layer (a tmpfs on some hosts — i.e. RAM) while the job dir is
    # the mounted data volume that actually has the space.
    with tempfile.TemporaryDirectory(prefix=".clippyme-upscale-",
                                     dir=os.path.dirname(os.path.abspath(output_path))) as tmp:
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

    # Pre-spend disk check: the staged PNG frames dwarf the clip itself, and
    # running out mid-render wastes the whole (slow) pass and leaves ffmpeg
    # failing on a half-written frame set. Fail fast with the real numbers
    # instead — same "reject before spending" posture as pipeline/preflight.py.
    frame_count = max(1, int(probe_duration(input_path) * _probe_fps_float(input_path)))
    needed = estimate_peak_bytes(width, height, frame_count, scale)
    stage_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    try:
        free = shutil.disk_usage(stage_dir).free
    except OSError:
        free = None
    if free is not None and free < needed:
        raise ConflictError(
            f"Not enough free disk to upscale this clip: needs about "
            f"{needed / 1e9:.1f} GB of temporary frames, {free / 1e9:.1f} GB free. "
            "Free up space, or upscale a shorter clip."
        )

    upscale_video(input_path, output_path, scale=scale)
    return {"scale": scale, "width": width * scale, "height": height * scale}
