"""On-demand provisioning for the Real-ESRGAN ncnn-vulkan binary.

Real-ESRGAN (github.com/xinntao/Real-ESRGAN-ncnn-vulkan) is a free,
open-source, self-hosted AI super-resolution tool — it is what
``domain.upscale`` shells out to for the "Upscale to 4K" clip action. It ships
as a standalone Vulkan-backed CLI binary + a ``models/`` folder, distributed as
a platform zip on GitHub Releases; it is NOT a pip dependency, mirroring how
the auto-editor binary is handled (see ``integrations.auto_editor_updater``).

Unlike auto-editor, ClippyMe does not bake a version-pinned copy into the
Docker image at build time (the release has no versioned pip-style artifact
registry to pin confidently against, and upscaling is an opt-in, per-clip
action rather than a core pipeline dependency). Instead this module fetches
and verifies it LAZILY the first time a user actually requests an upscale,
then caches it under ``/app/data/bin/realesrgan`` (same writable, PATH-
adjacent volume the auto-editor updater already uses) so every later request
reuses the cached copy — no repeat downloads, no image rebuild required.

Security posture mirrors auto_editor_updater.py exactly: the GitHub *release
metadata* fetch uses a no-redirect opener (SSRF hardening), the asset
*download* follows redirects only to an allow-listed CDN host, the archive is
sha256-verified against the digest GitHub publishes for the asset BEFORE
anything inside it is trusted, and only after that does it get extracted,
chmod +x'd and sanity-invoked. A release whose asset carries no published
digest is refused, never silently trusted — operators can always place a
self-vetted binary at ``BIN_DIR`` (or anywhere on PATH) themselves and this
module will pick it up instead of ever touching the network.
"""
import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from typing import Optional
from urllib.parse import urlparse
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

logger = logging.getLogger(__name__)

GITHUB_LATEST_API = (
    "https://api.github.com/repos/xinntao/Real-ESRGAN-ncnn-vulkan/releases/latest"
)

BIN_DIR = "/app/data/bin/realesrgan"
BINARY_NAME = "realesrgan-ncnn-vulkan"

HTTP_TIMEOUT = 15
MAX_RELEASE_JSON_BYTES = 2 * 1024 * 1024
# The release zip bundles the binary + several model .bin/.param files —
# comfortably under 200 MB even with all models included; this just stops a
# compromised/hijacked mirror from streaming an unbounded payload.
MAX_ZIP_BYTES = 300 * 1024 * 1024

_EXEC_MAGICS = (b"\x7fELF",)  # Docker images are Linux-only — ELF is the only accepted format.

_ALLOWED_DOWNLOAD_HOSTS = frozenset({
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
})


def auto_download_enabled() -> bool:
    """Whether provisioning may fetch the binary from GitHub on first use.

    ON by default: nothing downloads until a user actually clicks "Upscale to
    4K" (the endpoint is opt-in per request), so there is no always-on
    background network activity to gate the way auto-editor's daily updater
    needs to be. Set to 0 for a fully offline/pinned deployment — in that case
    place the binary + its models/ folder at ``BIN_DIR`` (or anywhere on
    PATH) yourself and it is used exactly the same way.
    """
    return os.environ.get("CLIPPYME_UPSCALE_AUTO_DOWNLOAD", "1") == "1"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects on the GitHub API JSON fetch — see auto_editor_updater."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class _SafeAssetRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed_download_url(newurl):
            raise RuntimeError("Real-ESRGAN asset redirected to an untrusted host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_API_OPENER = urllib.request.build_opener(_NoRedirectHandler)
_ASSET_OPENER = urllib.request.build_opener(_SafeAssetRedirectHandler)


def _allowed_download_url(url: str) -> bool:
    try:
        parsed = urlparse((url or "").strip())
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() in _ALLOWED_DOWNLOAD_HOSTS
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
        and not parsed.fragment
    )


def _detect_asset_substring() -> Optional[str]:
    """A distinctive substring of the release asset name for this platform.

    Only Linux x86_64 is supported today — that is the only architecture the
    upstream project publishes a prebuilt Vulkan binary for. Anything else
    (including aarch64 Docker images) returns None so the caller can surface
    a clear "unsupported platform" error instead of a confusing download
    failure.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "ubuntu"
    return None


def resolve_binary() -> Optional[str]:
    """Return a usable ``realesrgan-ncnn-vulkan`` path, or None if not present.

    Checks PATH first (an operator-installed / custom-image binary is always
    trusted over anything this module would fetch itself), then the
    auto-download cache dir. Never touches the network.
    """
    on_path = shutil.which(BINARY_NAME)
    if on_path:
        return on_path
    candidate = os.path.join(BIN_DIR, BINARY_NAME)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _fetch_latest_release() -> Optional[dict]:
    try:
        req = urllib.request.Request(
            GITHUB_LATEST_API,
            headers={"User-Agent": "ClippyMe-RealESRGANProvisioner/1.0"},
        )
        with _API_OPENER.open(req, timeout=HTTP_TIMEOUT) as resp:  # nosec B310: fixed official API URL, redirects disabled
            raw = resp.read(MAX_RELEASE_JSON_BYTES + 1)
        if len(raw) > MAX_RELEASE_JSON_BYTES:
            raise ValueError("GitHub release response exceeded size cap")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("GitHub release response was not an object")
    except Exception as e:
        logger.warning("Real-ESRGAN provisioner: GitHub API check failed: %s", e)
        return None

    asset_list = data.get("assets") or []
    if not isinstance(asset_list, list):
        return None
    assets = []
    for asset in asset_list:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name")
        if not isinstance(name, str) or not name:
            continue
        url = asset.get("browser_download_url")
        digest = asset.get("digest")
        assets.append({
            "name": name,
            "url": url if isinstance(url, str) else None,
            "digest": digest if isinstance(digest, str) else None,
        })
    return {"tag": data.get("tag_name"), "assets": assets}


def _verify_digest(path: str, expected_digest: Optional[str]) -> Optional[bool]:
    if not expected_digest:
        return None
    algo, _, want = expected_digest.partition(":")
    want = want.strip().lower()
    if algo.lower() != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", want):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == want


def _download_zip(url: str, target_path: str, expected_digest: Optional[str]) -> bool:
    if not _allowed_download_url(url):
        logger.warning("Real-ESRGAN provisioner: refusing untrusted download URL: %r", url)
        return False
    if not expected_digest:
        logger.error(
            "Real-ESRGAN provisioner: release asset has no published SHA256 digest; "
            "refusing to auto-download. Install the binary manually at %s instead.",
            BIN_DIR,
        )
        return False

    req = urllib.request.Request(url, headers={"User-Agent": "ClippyMe-RealESRGANProvisioner/1.0"})
    total = 0
    with _ASSET_OPENER.open(req, timeout=HTTP_TIMEOUT * 4) as resp:  # nosec B310: HTTPS host + redirects allow-listed
        with open(target_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ZIP_BYTES:
                    raise RuntimeError("download exceeded size cap")
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

    verdict = _verify_digest(target_path, expected_digest)
    if verdict is not True:
        logger.error("Real-ESRGAN provisioner: missing/invalid SHA256 or digest mismatch — rejecting asset")
        return False
    return True


def _extract_and_stage(zip_path: str, stage_dir: str) -> Optional[str]:
    """Extract the zip into ``stage_dir`` and return the binary path inside it.

    Guards against zip-slip (entries escaping ``stage_dir``) before writing
    anything. Returns None if no matching binary is found inside.
    """
    with zipfile.ZipFile(zip_path) as zf:
        stage_abs = os.path.abspath(stage_dir)
        for info in zf.infolist():
            dest = os.path.abspath(os.path.join(stage_dir, info.filename))
            if not (dest == stage_abs or dest.startswith(stage_abs + os.sep)):
                raise RuntimeError(f"Real-ESRGAN archive entry escapes staging dir: {info.filename!r}")
        zf.extractall(stage_dir)

    for root, _dirs, files in os.walk(stage_dir):
        if BINARY_NAME in files:
            return os.path.join(root, BINARY_NAME)
    return None


_PROVISION_LOCK = threading.Lock()


@contextlib.contextmanager
def _provision_lock():
    """Best-effort exclusive lock so concurrent upscale requests don't race a download."""
    if fcntl is None:
        acquired = _PROVISION_LOCK.acquire(blocking=True, timeout=600)
        try:
            yield acquired
        finally:
            if acquired:
                _PROVISION_LOCK.release()
        return

    os.makedirs(os.path.dirname(BIN_DIR), exist_ok=True)
    lock_path = BIN_DIR + ".lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield True
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def provision(*, log_fn=logger.info) -> Optional[str]:
    """Download, verify, extract and cache the binary. Returns its path or None.

    Safe to call repeatedly/concurrently — the second caller in wins the lock
    and finds ``resolve_binary()`` already satisfied by the first.
    """
    already = resolve_binary()
    if already:
        return already

    asset_substring = _detect_asset_substring()
    if asset_substring is None:
        log_fn(
            f"Real-ESRGAN: unsupported platform {platform.system()}/{platform.machine()} "
            "for auto-download — install the binary + models/ folder manually at "
            f"{BIN_DIR} to enable 4K upscaling."
        )
        return None

    release = _fetch_latest_release()
    if release is None:
        log_fn("Real-ESRGAN: could not reach GitHub releases API — upscaling unavailable")
        return None

    match = next(
        (a for a in release["assets"] if asset_substring in a["name"].lower() and a["name"].lower().endswith(".zip")),
        None,
    )
    if match is None or not match["url"]:
        log_fn(f"Real-ESRGAN: no matching release asset found for {asset_substring} in {release.get('tag')}")
        return None

    with _provision_lock() as acquired:
        if not acquired:
            log_fn("Real-ESRGAN: another worker holds the provisioning lock; skipping")
            return resolve_binary()

        already = resolve_binary()
        if already:
            return already

        with tempfile.TemporaryDirectory(prefix="realesrgan-provision-") as tmp:
            zip_path = os.path.join(tmp, "release.zip")
            try:
                ok = _download_zip(match["url"], zip_path, match["digest"])
            except Exception as e:
                log_fn(f"Real-ESRGAN: download failed: {e}")
                return None
            if not ok:
                return None

            stage_dir = os.path.join(tmp, "stage")
            os.makedirs(stage_dir, exist_ok=True)
            try:
                binary_path = _extract_and_stage(zip_path, stage_dir)
            except Exception as e:
                log_fn(f"Real-ESRGAN: archive extraction failed: {e}")
                return None
            if binary_path is None:
                log_fn(f"Real-ESRGAN: {BINARY_NAME} not found inside the downloaded archive")
                return None

            with open(binary_path, "rb") as f:
                head = f.read(4)
            if not any(head.startswith(m) for m in _EXEC_MAGICS):
                log_fn(f"Real-ESRGAN: downloaded file is not an ELF executable (magic={head!r})")
                return None
            os.chmod(binary_path, 0o700)

            try:
                subprocess.run([binary_path, "-h"], capture_output=True, timeout=10)
            except Exception as e:
                log_fn(f"Real-ESRGAN: sanity invocation failed: {e}")
                return None

            payload_root = os.path.dirname(binary_path)
            os.makedirs(os.path.dirname(BIN_DIR), exist_ok=True)
            if os.path.isdir(BIN_DIR):
                shutil.rmtree(BIN_DIR, ignore_errors=True)
            shutil.move(payload_root, BIN_DIR)

        final_path = resolve_binary()
        if final_path:
            log_fn(f"Real-ESRGAN: provisioned {release.get('tag')} at {final_path}")
        return final_path


def ensure_binary(*, log_fn=logger.info) -> Optional[str]:
    """Resolve an existing binary, else auto-provision one if enabled."""
    existing = resolve_binary()
    if existing:
        return existing
    if not auto_download_enabled():
        return None
    return provision(log_fn=log_fn)
