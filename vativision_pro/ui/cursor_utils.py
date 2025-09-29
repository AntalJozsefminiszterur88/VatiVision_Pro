"""Helper utilities for preparing cursor graphics."""

from __future__ import annotations

from PySide6 import QtGui


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
