"""Post-hoc 4K upscale orchestration for POST /api/upscale/{job_id}/{clip_index}.

Mirrors ``reframe_service.run_reframe``'s shape (resolve → lock → render →
atomically replace → persist metadata → update in-memory job state) but the
render itself runs in-process via ``domain.upscale`` instead of spawning a
``main.py`` subprocess — there is no cv2/torch import to isolate here, just
ffmpeg + the Real-ESRGAN binary.
"""
import asyncio
import logging
import os
import time

from clippyme.domain import upscale
from clippyme.domain.clip_locks import clip_lock
from clippyme.domain.clip_resolve import resolve_clip
from clippyme.domain.errors import ClippyMeError, ValidationError
from clippyme.domain.job_artifacts import save_job_metadata
from clippyme.pipeline.media_probe import probe_duration

logger = logging.getLogger(__name__)


async def run_upscale(*, job_id: str, clip_index: int, output_root: str,
                      jobs: dict, scale: int | None = None) -> dict:
    """Re-render one clip at 4K in place. Returns the endpoint response payload
    (cache-busted ``new_video_url`` + the resolved scale/resolution)."""
    resolved = resolve_clip(job_id, clip_index, output_root)

    duration = await asyncio.to_thread(probe_duration, resolved.clip_path)
    max_duration = upscale.max_clip_duration_seconds()
    if duration > max_duration:
        raise ValidationError(
            f"Clip is {duration:.0f}s — 4K upscaling is capped at {max_duration}s "
            "on this server (CLIPPYME_UPSCALE_MAX_DURATION_SECONDS)."
        )

    async with clip_lock(resolved.job_dir, clip_index):
        tmp_path = resolved.clip_path + ".upscale.tmp.mp4"
        try:
            result = await asyncio.to_thread(
                upscale.upscale_clip_to_4k, resolved.clip_path, tmp_path,
                requested_scale=scale,
            )
            os.replace(tmp_path, resolved.clip_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Cache-busting suffix only in the HTTP response — never in the stored
        # video_url, which every clip-file consumer resolves verbatim (see
        # reframe_service for why a trailing ?v= there breaks publish/compose).
        cache_bust = int(time.time())
        clean_video_url = f"/videos/{job_id}/{resolved.clip_filename}"
        new_video_url = f"{clean_video_url}?v={cache_bust}"

        clips = resolved.metadata.get("shorts", [])
        clips[clip_index]["video_url"] = clean_video_url
        clips[clip_index]["upscaled_4k"] = True
        clips[clip_index]["upscale_scale"] = result["scale"]
        clips[clip_index]["resolution"] = f"{result['width']}x{result['height']}"
        resolved.metadata["shorts"] = clips

        save_failed = None
        try:
            save_job_metadata(resolved.metadata_path, resolved.metadata)
        except Exception as e:
            logger.error("Failed to persist metadata.json after upscale: %s", e)
            save_failed = e

        if (
            job_id in jobs
            and "result" in jobs[job_id]
            and "clips" in jobs[job_id]["result"]
            and clip_index < len(jobs[job_id]["result"]["clips"])
        ):
            live_clip = jobs[job_id]["result"]["clips"][clip_index]
            live_clip["video_url"] = clean_video_url
            live_clip["upscaled_4k"] = True
            live_clip["upscale_scale"] = result["scale"]
            live_clip["resolution"] = f"{result['width']}x{result['height']}"

        if save_failed is not None:
            raise ClippyMeError(
                "Upscale succeeded but metadata persistence failed; reload may show stale state",
                status_code=500,
            )

    return {
        "success": True,
        "new_video_url": new_video_url,
        "scale": result["scale"],
        "resolution": f"{result['width']}x{result['height']}",
    }
