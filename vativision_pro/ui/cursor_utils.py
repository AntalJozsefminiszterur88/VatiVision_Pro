"""Helper utilities for preparing cursor graphics."""

from __future__ import annotations

import logging
import sys
from typing import Optional, Tuple

from PySide6 import QtGui

logger = logging.getLogger(__name__)


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


def get_system_cursor_pixmap() -> Tuple[Optional[QtGui.QPixmap], Optional[int]]:
    """Return the currently active system cursor pixmap on Windows.

    The function returns a tuple containing the sanitized pixmap (if one could be
    obtained) and the numeric value of the cursor handle.  The handle value can be
    used by callers to detect shape changes without comparing pixmaps directly.
    On non-Windows platforms the function simply returns ``(None, None)``.
    """

    if not sys.platform.startswith("win"):
        return None, None

    try:
        import ctypes
        from ctypes import wintypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        class CURSORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", wintypes.HCURSOR),
                ("ptScreenPos", POINT),
            ]

        CURSOR_SHOWING = 0x00000001

        user32 = ctypes.windll.user32
        info = CURSORINFO()
        info.cbSize = ctypes.sizeof(CURSORINFO)

        if not user32.GetCursorInfo(ctypes.byref(info)):
            return None, None

        handle_value = ctypes.cast(info.hCursor, ctypes.c_void_p).value or 0
        handle_int = int(handle_value)

        if not (info.flags & CURSOR_SHOWING):
            return None, handle_int

        hicon = user32.CopyIcon(info.hCursor)
        if not hicon:
            return None, handle_int

        try:
            pixmap = QtGui.QPixmap.fromWinHICON(hicon)
        finally:
            user32.DestroyIcon(hicon)

        if pixmap.isNull():
            return None, handle_int

        return sanitize_cursor_pixmap(pixmap), handle_int
    except Exception:  # pragma: no cover - platform specific defensive guard
        logger.exception("Failed to query system cursor pixmap.")
        return None, None
