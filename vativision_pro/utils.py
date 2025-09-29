"""Utility helpers for VatiVision Pro."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _resolve_documents_dir() -> Path:
    """Return the user's documents directory in a locale agnostic way."""

    override = os.getenv("VATIVISION_LOG_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform.startswith("win"):
        try:
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            SHGFP_TYPE_CURRENT = 0
            CSIDL_PERSONAL = 5  # Documents folder
            if (
                ctypes.windll.shell32.SHGetFolderPathW(
                    None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf
                )
                == 0
            ):
                documents_dir = Path(buf.value)
                if documents_dir:
                    return documents_dir
        except Exception:  # pragma: no cover - best effort on Windows only
            logging.getLogger(__name__).debug(
                "Nem sikerült lekérdezni a Dokumentumok mappát a Windows API-ból.",
                exc_info=True,
            )

    home = Path.home()
    candidates = [
        home / "Documents",
        home / "Dokumentumok",
        home / "dokumentumok",
        home / "OneDrive" / "Documents",
        home / "OneDrive" / "Dokumentumok",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return home / "Documents"


def _setup_file_logging(log_root: Path) -> Path:
    """Initialise rotating file logging limited to 5 MB in the requested folder."""

    log_dir = log_root
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Nem sikerült létrehozni a log könyvtárat: {log_dir}") from exc

    log_path = log_dir / "vativision_pro.log"

    root_logger = logging.getLogger()
    handler_exists = False
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == str(log_path):
            handler_exists = True
            break

    if not handler_exists:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.INFO)

    logging.captureWarnings(True)
    return log_path
