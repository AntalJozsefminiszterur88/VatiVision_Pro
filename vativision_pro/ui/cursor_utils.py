"""Helper utilities for preparing cursor graphics."""

from __future__ import annotations

import logging
import sys
from typing import Dict, Optional, Tuple

from PySide6 import QtGui

logger = logging.getLogger(__name__)


CursorShape = Optional[str]
_STANDARD_CURSOR_HANDLE_CACHE: Dict[int, Optional[int]] = {}


def _is_close(color: QtGui.QColor, reference: QtGui.QColor, tolerance: int) -> bool:
    return (
        abs(color.red() - reference.red()) <= tolerance
        and abs(color.green() - reference.green()) <= tolerance
        and abs(color.blue() - reference.blue()) <= tolerance
    )


def sanitize_cursor_pixmap(pixmap: QtGui.QPixmap, *, tolerance: int = 12) -> QtGui.QPixmap:
    """Return a pixmap without opaque background or dark silhouettes."""

    if pixmap.isNull():
        return pixmap

    image = pixmap.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        return pixmap

    background = image.pixelColor(0, 0)
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() == 0:
                continue

            make_transparent = False
            if _is_close(color, background, tolerance):
                make_transparent = True
            else:
                if color.alpha() < 220 and max(color.red(), color.green(), color.blue()) < 40:
                    make_transparent = True

            if make_transparent:
                color.setAlpha(0)
                color.setRed(0)
                color.setGreen(0)
                color.setBlue(0)
                image.setPixelColor(x, y, color)

    return QtGui.QPixmap.fromImage(image)


def get_system_cursor_pixmap() -> Tuple[Optional[QtGui.QPixmap], Optional[int], CursorShape]:
    """Return the current system cursor pixmap, handle and shape on Windows.

    The function returns a tuple containing the sanitized pixmap (if one could be
    obtained), the numeric value of the cursor handle, and a symbolic cursor
    shape identifier.  When the cursor matches one of the standard Windows
    cursors we resolve the handle to ``"arrow"``, ``"ibeam"`` or ``"link"``.
    On non-Windows platforms the function simply returns ``(None, None, None)``.
    """

    if not sys.platform.startswith("win"):
        return None, None, None

    try:
        import ctypes
        from ctypes import wintypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        HCURSOR = getattr(wintypes, "HCURSOR", ctypes.c_void_p)

        class CURSORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", HCURSOR),
                ("ptScreenPos", POINT),
            ]

        CURSOR_SHOWING = 0x00000001

        user32 = ctypes.windll.user32
        user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
        user32.GetCursorInfo.restype = wintypes.BOOL
        user32.CopyIcon.argtypes = [HCURSOR]
        user32.CopyIcon.restype = ctypes.c_void_p
        user32.DestroyIcon.argtypes = [HCURSOR]
        user32.DestroyIcon.restype = wintypes.BOOL
        user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.LoadCursorW.restype = ctypes.c_void_p

        info = CURSORINFO()
        info.cbSize = ctypes.sizeof(CURSORINFO)

        if not user32.GetCursorInfo(ctypes.byref(info)):
            return None, None, None

        handle_value = ctypes.cast(info.hCursor, ctypes.c_void_p).value or 0
        handle_int = int(handle_value)
        shape = _detect_cursor_shape(user32, handle_int)

        if not (info.flags & CURSOR_SHOWING):
            return None, handle_int, shape

        hicon = user32.CopyIcon(info.hCursor)
        if not hicon:
            return None, handle_int, shape

        try:
            pixmap = QtGui.QPixmap.fromWinHICON(hicon)
        finally:
            user32.DestroyIcon(hicon)

        if pixmap.isNull():
            return None, handle_int, shape

        return sanitize_cursor_pixmap(pixmap), handle_int, shape
    except Exception:  # pragma: no cover - platform specific defensive guard
        logger.exception("Failed to query system cursor pixmap.")
        return None, None, None


def _load_standard_cursor_handle(user32, resource_id: int) -> Optional[int]:
    """Return the shared handle for a well-known cursor resource."""

    try:
        import ctypes
    except Exception:  # pragma: no cover - defensive guard
        return None

    cached = _STANDARD_CURSOR_HANDLE_CACHE.get(resource_id)
    if cached is not None:
        return cached

    try:
        handle = user32.LoadCursorW(None, ctypes.c_void_p(resource_id))
    except Exception:  # pragma: no cover - defensive guard
        cached = None
    else:
        if not handle:
            cached = None
        else:
            cached = int(ctypes.cast(handle, ctypes.c_void_p).value or 0)

    _STANDARD_CURSOR_HANDLE_CACHE[resource_id] = cached
    return cached


def _detect_cursor_shape(user32, handle_int: int) -> CursorShape:
    """Map a cursor handle to a symbolic Windows cursor shape if possible."""

    if not handle_int:
        return None

    try:
        import ctypes  # noqa: WPS433 - used for pointer comparison
    except Exception:  # pragma: no cover - defensive guard
        return None

    mapping = {
        "arrow": 32512,  # IDC_ARROW
        "ibeam": 32513,  # IDC_IBEAM
        "link": 32649,  # IDC_HAND
    }

    for shape, resource_id in mapping.items():
        std_handle = _load_standard_cursor_handle(user32, resource_id)
        if std_handle and std_handle == handle_int:
            return shape

    return None
