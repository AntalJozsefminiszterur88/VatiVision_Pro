"""Configuration constants for the VatiVision Pro application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .utils import _resolve_documents_dir, _setup_file_logging


def _iter_project_roots() -> list[Path]:
    """Return possible base directories for bundled assets.

    When the application is packaged with PyInstaller the resources are
    temporarily extracted either next to the executable or into the
    ``_MEIPASS`` directory.  During development ``__file__`` still points to
    the repository checkout.  We return every relevant candidate so that the
    caller can select the first existing path.
    """

    roots = [Path(__file__).resolve().parent.parent]

    if getattr(sys, "frozen", False):  # running from a packaged executable
        executable_dir = Path(sys.executable).resolve().parent
        if executable_dir not in roots:
            roots.insert(0, executable_dir)

        bundle_dir = Path(getattr(sys, "_MEIPASS", executable_dir))
        if bundle_dir not in roots:
            roots.insert(0, bundle_dir)

    return roots


def _resolve_project_root() -> Path:
    """Select the most suitable base directory for bundled assets."""

    for root in _iter_project_roots():
        package_dir = root / "vativision_pro"
        media_dir = package_dir / "media"
        if media_dir.exists():
            return root
        if package_dir.exists():
            return root

    # Fallback to the repository checkout when nothing else matches.
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _resolve_project_root()
MEDIA_DIR = PROJECT_ROOT / "vativision_pro" / "media"
BONECA_AMBALABU_AUDIO = MEDIA_DIR / "boneca ambalabu.mp3"

SIGNALING_WS = "wss://vatib-vezerlo.duckdns.org/ws"
TURN_HOST = "vatib-vezerlo.duckdns.org"
ROOM_NAME = "VATI-ROOM"
ROOM_PIN = "428913"
STUN_URLS = ["stun:stun.l.google.com:19302"]

APP_TITLE = "VatiVision Pro - UMKGL Solutions"
def _resolve_app_icon_path() -> Path:
    """Locate the bundled program logo regardless of the execution mode."""

    for root in _iter_project_roots():
        candidate = root / "program_logo.png"
        if candidate.exists():
            return candidate
    # Fall back to the repository location; the file may simply be missing.
    return Path(__file__).resolve().parent.parent / "program_logo.png"


APP_ICON_PATH = _resolve_app_icon_path()
CURSOR_IMAGE_PATH = MEDIA_DIR / "cursor.png"

BW_SECONDS = 5.0
BW_CHUNK = 16 * 1024
BW_MAX_BUF = 4 * 1024 * 1024

DARK_BG = "#2b2d31"
DARK_CARD = "#313338"
DARK_ELEV = "#1e1f22"
ACCENT = "#5865f2"
TEXT = "#e3e5e8"

LOG_ROOT = _resolve_documents_dir() / "UMKGL Solutions" / "VatiVision_Pro"
LOG_FILE = _setup_file_logging(LOG_ROOT)

logger = logging.getLogger(__name__)
logger.info("Fájllog inicializálva: %s", LOG_FILE)

from vativision_pro.media.screenshare import RESOLUTIONS  # noqa: E402  (circular guard)

__all__ = [
    "PROJECT_ROOT",
    "MEDIA_DIR",
    "SIGNALING_WS",
    "TURN_HOST",
    "ROOM_NAME",
    "ROOM_PIN",
    "STUN_URLS",
    "APP_TITLE",
    "APP_ICON_PATH",
    "CURSOR_IMAGE_PATH",
    "BONECA_AMBALABU_AUDIO",
    "BW_SECONDS",
    "BW_CHUNK",
    "BW_MAX_BUF",
    "DARK_BG",
    "DARK_CARD",
    "DARK_ELEV",
    "ACCENT",
    "TEXT",
    "LOG_ROOT",
    "LOG_FILE",
    "RESOLUTIONS",
]
