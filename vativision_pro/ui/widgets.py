"""Custom Qt widget implementations for VatiVision Pro."""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..media.screenshare import (
    CURSOR_SCALE_RATIO,
    CURSOR_MIN_SIZE,
    CURSOR_OFFSET_X,
    CURSOR_OFFSET_Y,
)
from .cursor_utils import sanitize_cursor_pixmap

class AnimatedButton(QtWidgets.QPushButton):
    """QPushButton subclass that animates a subtle highlight when pressed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._effect = QtWidgets.QGraphicsColorizeEffect(self)
        self._effect.setColor(QtGui.QColor("#ffffff"))
        self._effect.setStrength(0.0)
        self.setGraphicsEffect(self._effect)

        self._anim = QtCore.QPropertyAnimation(self._effect, b"strength", self)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        self.pressed.connect(self._on_pressed)
        self.released.connect(self._on_released)

    def _animate_strength(self, target: float, duration: int):
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(self._effect.strength())
        self._anim.setEndValue(target)
        self._anim.start()

    def _on_pressed(self):
        self._animate_strength(0.45, 110)

    def _on_released(self):
        self._animate_strength(0.0, 180)


class VideoSurface(QtWidgets.QLabel):
    """Clickable video surface that emits normalized coordinates."""

    clicked = QtCore.Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap_size = QtCore.QSize()

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:  # type: ignore[override]
        super().setPixmap(pixmap)
        if pixmap is None or pixmap.isNull():
            self._pixmap_size = QtCore.QSize()
        else:
            self._pixmap_size = pixmap.size()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._emit_click(event.position())
        super().mousePressEvent(event)

    def _emit_click(self, pos: QtCore.QPointF) -> None:
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return
        label_w = self.width()
        label_h = self.height()
        offset_x = max(0.0, (label_w - pix_w) / 2.0)
        offset_y = max(0.0, (label_h - pix_h) / 2.0)
        local_x = pos.x() - offset_x
        local_y = pos.y() - offset_y
        if local_x < 0 or local_y < 0 or local_x > pix_w or local_y > pix_h:
            return
        norm_x = float(local_x / pix_w)
        norm_y = float(local_y / pix_h)
        self.clicked.emit(norm_x, norm_y)


class FullscreenViewer(QtWidgets.QWidget):
    """Window that displays the shared video feed in fullscreen."""

    closed = QtCore.Signal()
    clicked = QtCore.Signal(float, float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("VatiVision Pro — Teljes képernyő")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)

        self._pixmap: Optional[QtGui.QPixmap] = None
        self._cursor_pixmap = QtGui.QPixmap()
        self._cursor_scaled: Optional[QtGui.QPixmap] = None
        self._cursor_scaled_width: int = 0
        self._pointer_norm: Optional[Tuple[float, float]] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = VideoSurface()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.label.clicked.connect(self.clicked.emit)
        layout.addWidget(self.label, 1)

        self.pointer_overlay = QtWidgets.QLabel(self.label)
        self.pointer_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.pointer_overlay.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.pointer_overlay.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.pointer_overlay.setStyleSheet("background: transparent;")
        self.pointer_overlay.hide()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.hide_pointer_overlay()
        self.closed.emit()
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._update_scaled_pixmap()
        self._reposition_pointer_overlay()
        super().resizeEvent(event)

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._update_scaled_pixmap()
        self._reposition_pointer_overlay()

    def set_cursor_source(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap and not pixmap.isNull():
            self._cursor_pixmap = sanitize_cursor_pixmap(pixmap)
        else:
            self._cursor_pixmap = QtGui.QPixmap()
        self._cursor_scaled = None
        self._cursor_scaled_width = 0
        if self._pointer_norm is not None and self.pointer_overlay.isVisible():
            nx, ny = self._pointer_norm
            self._show_pointer_overlay(nx, ny)
            self._reposition_pointer_overlay()

    def update_pointer(self, norm_x: float, norm_y: float, visible: bool) -> None:
        if not visible:
            self.hide_pointer_overlay()
            return
        nx = max(0.0, min(1.0, float(norm_x)))
        ny = max(0.0, min(1.0, float(norm_y)))
        if not self._show_pointer_overlay(nx, ny):
            self._pointer_norm = (nx, ny)
            return
        self._pointer_norm = (nx, ny)
        self._reposition_pointer_overlay()

    def _update_scaled_pixmap(self) -> None:
        if not self._pixmap:
            self.label.clear()
            return
        if self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)
        if self._pointer_norm is not None:
            nx, ny = self._pointer_norm
            if not self.pointer_overlay.isVisible():
                self._show_pointer_overlay(nx, ny)
            self._reposition_pointer_overlay()

    def _prepare_cursor_pixmap(self) -> Optional[QtGui.QPixmap]:
        if self._cursor_pixmap.isNull():
            return None
        pixmap = self.label.pixmap()
        if not pixmap or pixmap.isNull():
            return None
        target_width = max(
            CURSOR_MIN_SIZE, int(round(pixmap.width() * CURSOR_SCALE_RATIO))
        )
        target_width = min(target_width, self._cursor_pixmap.width())
        if target_width <= 0:
            return None
        if self._cursor_scaled is None or self._cursor_scaled_width != target_width:
            self._cursor_scaled = self._cursor_pixmap.scaledToWidth(
                target_width, QtCore.Qt.SmoothTransformation
            )
            self._cursor_scaled_width = self._cursor_scaled.width()
        return self._cursor_scaled

    def _show_pointer_overlay(self, norm_x: float, norm_y: float) -> bool:
        pix = self._prepare_cursor_pixmap()
        if pix is None:
            return False
        self.pointer_overlay.setPixmap(pix)
        self.pointer_overlay.resize(pix.size())
        self.pointer_overlay.show()
        self._pointer_norm = (norm_x, norm_y)
        return True

    def _reposition_pointer_overlay(self) -> None:
        if self._pointer_norm is None or not self.pointer_overlay.isVisible():
            return
        pixmap = self.label.pixmap()
        if not pixmap or pixmap.isNull():
            self.hide_pointer_overlay()
            return
        lw, lh = self.label.width(), self.label.height()
        pw, ph = pixmap.width(), pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        offset_x = max(0, (lw - pw) // 2)
        offset_y = max(0, (lh - ph) // 2)
        nx, ny = self._pointer_norm
        x = offset_x + int(round(nx * pw + CURSOR_OFFSET_X))
        y = offset_y + int(round(ny * ph + CURSOR_OFFSET_Y))
        ow = self.pointer_overlay.width()
        oh = self.pointer_overlay.height()
        x = max(0, min(x, lw - ow))
        y = max(0, min(y, lh - oh))
        self.pointer_overlay.move(x, y)

    def hide_pointer_overlay(self) -> None:
        self.pointer_overlay.hide()
        self._pointer_norm = None


class ScreenPointerOverlay(QtWidgets.QWidget):
    """Transparent overlay window displayed over the shared monitor."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        flags = (
            QtCore.Qt.Tool
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.WindowDoesNotAcceptFocus
        )
        super().__init__(parent, flags)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.hide()

        self._cursor_pixmap = QtGui.QPixmap()
        self._cursor_scaled: Optional[QtGui.QPixmap] = None
        self._cursor_scaled_width: int = 0
        self._pointer_norm: Optional[Tuple[float, float]] = None
        self._capture_bbox: Optional[Tuple[int, int, int, int]] = None

        self.label = QtWidgets.QLabel(self)
        self.label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.label.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label.setStyleSheet("background: transparent;")
        self.label.hide()

    def set_cursor_source(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap and not pixmap.isNull():
            self._cursor_pixmap = sanitize_cursor_pixmap(pixmap)
        else:
            self._cursor_pixmap = QtGui.QPixmap()
        self._cursor_scaled = None
        self._cursor_scaled_width = 0
        if self._pointer_norm and self._capture_bbox:
            nx, ny = self._pointer_norm
            self.show_pointer(nx, ny, self._capture_bbox)

    def hide_pointer(self) -> None:
        self._pointer_norm = None
        self._capture_bbox = None
        self.label.hide()
        self.hide()

    def show_pointer(
        self, norm_x: float, norm_y: float, bbox: Tuple[int, int, int, int]
    ) -> None:
        if not self._cursor_pixmap or self._cursor_pixmap.isNull():
            self.hide_pointer()
            return
        left, top, width, height = bbox
        if width <= 0 or height <= 0:
            self.hide_pointer()
            return
        pix = self._prepare_cursor_pixmap(width)
        if pix is None:
            self.hide_pointer()
            return

        self._pointer_norm = (
            max(0.0, min(1.0, float(norm_x))),
            max(0.0, min(1.0, float(norm_y))),
        )
        self._capture_bbox = bbox

        self.setGeometry(left, top, width, height)
        self.label.setPixmap(pix)
        self.label.resize(pix.size())

        nx, ny = self._pointer_norm
        cw, ch = pix.width(), pix.height()
        x = max(
            0,
            min(int(round(nx * width + CURSOR_OFFSET_X)), max(0, width - cw)),
        )
        y = max(
            0,
            min(int(round(ny * height + CURSOR_OFFSET_Y)), max(0, height - ch)),
        )
        self.label.move(x, y)
        self.label.show()
        if not self.isVisible():
            self.show()
        self.raise_()

    def _prepare_cursor_pixmap(self, capture_width: int) -> Optional[QtGui.QPixmap]:
        if self._cursor_pixmap.isNull() or capture_width <= 0:
            return None
        target_width = max(
            CURSOR_MIN_SIZE, int(round(capture_width * CURSOR_SCALE_RATIO))
        )
        target_width = min(target_width, self._cursor_pixmap.width())
        if target_width <= 0:
            return None
        if self._cursor_scaled is None or self._cursor_scaled_width != target_width:
            self._cursor_scaled = self._cursor_pixmap.scaledToWidth(
                target_width, QtCore.Qt.SmoothTransformation
            )
            self._cursor_scaled_width = self._cursor_scaled.width()
        return self._cursor_scaled
