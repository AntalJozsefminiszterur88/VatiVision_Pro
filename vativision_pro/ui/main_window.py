"""Main Qt window for the VatiVision Pro client."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from qasync import QEventLoop

from ..config import (
    APP_ICON_PATH,
    APP_TITLE,
    BW_CHUNK,
    BW_SECONDS,
    CURSOR_IMAGE_PATH,
    DARK_CARD,
    RESOLUTIONS,
    ROOM_NAME,
    ROOM_PIN,
    SIGNALING_WS,
)
from ..core import Core
from ..media.audio import audio_playback_supported, audio_capture_supported
from .style import style
from .widgets import AnimatedButton, VideoSurface, FullscreenViewer, ScreenPointerOverlay

class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 720)

        if APP_ICON_PATH.exists():
            self._app_icon = QtGui.QIcon(str(APP_ICON_PATH))
        else:
            self._app_icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        self.setWindowIcon(self._app_icon)

        self.ui_logger = logging.getLogger(f"{__name__}.UI")

        central = QtWidgets.QWidget(); self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        root = QtWidgets.QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        main_layout.addLayout(root, 1)

        side = QtWidgets.QFrame(); side.setFixedWidth(340)
        side.setStyleSheet(f"QFrame {{ background-color:{DARK_CARD}; border-radius:12px; }}")
        s = QtWidgets.QVBoxLayout(side); s.setContentsMargins(12,12,12,12); s.setSpacing(10)

        head = QtWidgets.QLabel(f"<b>VatiVision Pro - UMKGL Solutions</b><br>"
                                f"WS: <code>{SIGNALING_WS}</code><br>"
                                f"Szoba: <code>{ROOM_NAME}</code> &nbsp; PIN: <code>{ROOM_PIN}</code>")
        head.setTextFormat(QtCore.Qt.RichText)
        s.addWidget(head)

        self.status = QtWidgets.QLabel("Készenlét"); self.status.setObjectName("status"); self.status.setWordWrap(True)
        s.addWidget(self.status)

        self.chk_relay = QtWidgets.QCheckBox("Preferáld a TURN-öt (relay only)")
        s.addWidget(self.chk_relay)

        s.addWidget(QtWidgets.QLabel("Eseménynapló:"))
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(5000)
        s.addWidget(self.log, 1)

        root.addWidget(side)

        panel = QtWidgets.QGroupBox("Szerep kiválasztása és indítás")
        v = QtWidgets.QVBoxLayout(panel)

        row = QtWidgets.QHBoxLayout()
        self.role_combo = QtWidgets.QComboBox()
        self.role_combo.addItem("Küldő", "sender")
        self.role_combo.addItem("Fogadó", "receiver")
        row.addWidget(QtWidgets.QLabel("Szerep:")); row.addWidget(self.role_combo); row.addStretch(1)
        v.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_start = AnimatedButton("Kapcsolat indítása")
        self.btn_stop  = AnimatedButton("Leállítás")
        self.btn_ping  = AnimatedButton("Ping küldése (Küldő)")
        self.btn_bw    = AnimatedButton("Sávszél-teszt")
        row2.addWidget(self.btn_start); row2.addWidget(self.btn_stop); row2.addWidget(self.btn_ping); row2.addWidget(self.btn_bw); row2.addStretch(1)
        v.addLayout(row2)

        share = QtWidgets.QGroupBox("Képernyőmegosztás (küldő)")
        sh = QtWidgets.QGridLayout(share)
        self.share_group = share

        self.res_combo = QtWidgets.QComboBox()
        for label in RESOLUTIONS.keys():
            self.res_combo.addItem(label)
        self.res_combo.setCurrentIndex(3)

        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setRange(0, 60); self.fps_slider.setValue(30)
        self.fps_label = QtWidgets.QLabel("FPS: 30")

        self.br_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.br_slider.setRange(100, 50000); self.br_slider.setSingleStep(50); self.br_slider.setValue(4000)
        self.br_label = QtWidgets.QLabel("Bitráta: 4000 kbps")

        self.chk_share_audio = QtWidgets.QCheckBox("Hang megosztása")
        if not audio_capture_supported():
            self.chk_share_audio.setEnabled(False)
            self.chk_share_audio.setToolTip(
                "A rendszerhang megosztása ezen az eszközön nem támogatott."
            )

        self.btn_share_start = AnimatedButton("Megosztás indítása")
        self.btn_share_stop  = AnimatedButton("Megosztás leállítása")

        self._pending_fps: Optional[int] = None
        self._pending_bitrate: Optional[int] = None

        self._fps_debounce = QtCore.QTimer(self)
        self._fps_debounce.setSingleShot(True)
        self._fps_debounce.timeout.connect(self._apply_pending_fps)

        self._br_debounce = QtCore.QTimer(self)
        self._br_debounce.setSingleShot(True)
        self._br_debounce.timeout.connect(self._apply_pending_bitrate)

        sh.addWidget(QtWidgets.QLabel("Felbontás:"), 0, 0)
        sh.addWidget(self.res_combo, 0, 1, 1, 2)
        sh.addWidget(self.fps_label, 1, 0)
        sh.addWidget(self.fps_slider, 1, 1, 1, 2)
        sh.addWidget(self.br_label, 2, 0)
        sh.addWidget(self.br_slider, 2, 1, 1, 2)
        sh.addWidget(self.chk_share_audio, 3, 0, 1, 3)
        sh.addWidget(self.btn_share_start, 4, 1)
        sh.addWidget(self.btn_share_stop, 4, 2)
        v.addWidget(share)

        preview_box = QtWidgets.QGroupBox("Bejövő videó (fogadó)")
        pv = QtWidgets.QVBoxLayout(preview_box)

        video_container = QtWidgets.QWidget()
        vc_layout = QtWidgets.QVBoxLayout(video_container)
        vc_layout.setContentsMargins(0, 0, 0, 0)
        vc_layout.setSpacing(6)

        self.video_label = VideoSurface("Nincs bejövő videó")
        self.video_label.setMinimumHeight(280)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1e1f22; border: 1px solid #3a3c41; border-radius: 8px; }"
        )
        self.video_label.clicked.connect(self.on_video_clicked)
        vc_layout.addWidget(self.video_label, 1)

        self.pointer_overlay = QtWidgets.QLabel(self.video_label)
        self.pointer_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.pointer_overlay.hide()

        controls_row = QtWidgets.QHBoxLayout()
        self.audio_volume_label = QtWidgets.QLabel("Hang: 80%")
        self.audio_volume_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.audio_volume_slider.setRange(0, 100)
        self.audio_volume_slider.setValue(80)
        self.audio_volume_slider.setSingleStep(5)
        if not audio_playback_supported():
            self.audio_volume_slider.setEnabled(False)
            tip = "A hang lejátszása ezen a rendszeren nem támogatott."
            self.audio_volume_slider.setToolTip(tip)
            self.audio_volume_label.setToolTip(tip)
        controls_row.addWidget(self.audio_volume_label)
        controls_row.addWidget(self.audio_volume_slider, 2)
        controls_row.addStretch(1)
        self.btn_fullscreen = AnimatedButton("Teljes képernyő")
        controls_row.addWidget(self.btn_fullscreen)
        vc_layout.addLayout(controls_row)

        pv.addWidget(video_container)
        v.addWidget(preview_box, 1)

        self.inbox = QtWidgets.QPlainTextEdit(); self.inbox.setReadOnly(True)
        self.inbox.setPlaceholderText("Fogadó panel — bejövő üzenetek")
        v.addWidget(self.inbox, 1)

        root.addWidget(panel, 1)

        self.settings = QtCore.QSettings("UMKGL Solutions", "VatiVision_Pro")

        self.core: Optional[Core] = None
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_ping.clicked.connect(self.on_ping)
        self.btn_bw.clicked.connect(self.on_bw)
        self.btn_share_start.clicked.connect(self.on_share_start)
        self.btn_share_stop.clicked.connect(self.on_share_stop)
        self.res_combo.currentIndexChanged.connect(self.on_res_changed)
        self.fps_slider.valueChanged.connect(self.on_fps_changed)
        self.br_slider.valueChanged.connect(self.on_br_changed)
        self.chk_share_audio.toggled.connect(
            lambda checked: self._save_setting("ui/share_audio", checked)
        )
        self.audio_volume_slider.valueChanged.connect(self.on_audio_volume_changed)

        self.role_combo.currentIndexChanged.connect(self.on_role_changed)
        self.chk_relay.toggled.connect(
            lambda checked: self._save_setting("ui/prefer_relay", checked)
        )
        self.res_combo.currentTextChanged.connect(
            lambda value: self._save_setting("ui/resolution", value)
        )
        self.btn_fullscreen.clicked.connect(self.on_toggle_fullscreen)

        self._tray_icon: Optional[QtWidgets.QSystemTrayIcon] = None
        self._tray_message_shown = False
        self._allow_close = False
        if QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self._create_tray_icon()
        else:
            self.log_ui_message("A rendszer nem támogatja a tálca ikont, kilépéskor az alkalmazás bezárul.", logging.INFO)

        self.fullscreen_window: Optional[FullscreenViewer] = None
        self._last_pixmap: Optional[QtGui.QPixmap] = None
        if CURSOR_IMAGE_PATH.exists():
            self._cursor_pixmap = QtGui.QPixmap(str(CURSOR_IMAGE_PATH))
        else:
            self._cursor_pixmap = QtGui.QPixmap()
        self._cursor_scaled: Optional[QtGui.QPixmap] = None
        self._cursor_scaled_width: int = 0
        self._pointer_timer = QtCore.QTimer(self)
        self._pointer_timer.setSingleShot(True)
        self._pointer_timer.timeout.connect(self.on_pointer_timeout)
        self._pointer_norm: Optional[Tuple[float, float]] = None
        self._pointer_local = False
        self._screen_pointer_overlay = ScreenPointerOverlay()
        self._screen_pointer_overlay.set_cursor_source(self._cursor_pixmap)

        self._restore_settings()
        self._update_role_ui(self.role_combo.currentData() or "sender")

    @QtCore.Slot(str)
    def append_log_message(self, message: str) -> None:
        self.log.appendPlainText(message)

    def log_ui_message(self, message: str, level: int = logging.INFO) -> None:
        self.ui_logger.log(level, message)
        self.log.appendPlainText(message)

    def _apply_pending_fps(self) -> None:
        value = self._pending_fps
        self._pending_fps = None
        if value is None:
            return
        if self.core:
            self.log_ui_message(f"FPS módosítása: {value}")
            asyncio.create_task(self.core.set_fps(value))

    def _apply_pending_bitrate(self) -> None:
        value = self._pending_bitrate
        self._pending_bitrate = None
        if value is None:
            return
        if self.core:
            self.log_ui_message(f"Bitráta módosítása: {value} kbps")
            asyncio.create_task(self.core.set_bitrate(value))

    @QtCore.Slot()
    def on_start(self):
        if self.core: return
        role_label = self.role_combo.currentText()
        role_value = self.role_combo.currentData() or "sender"
        prefer = self.chk_relay.isChecked()
        self.log_ui_message(f"Kapcsolat indítása szerep={role_label}, prefer_relay={prefer}")
        self.core = Core(role=role_value, prefer_relay=prefer)
        self.core.status.connect(self.status.setText)
        self.core.log.connect(self.append_log_message)
        self.core.msg_in.connect(self.inbox.appendPlainText)
        self.core.video_frame.connect(self.update_video)
        self.core.pointer.connect(self.on_pointer_update)
        self.core.set_audio_volume(self.audio_volume_slider.value() / 100.0)

        async def _run():
            try: await self.core.start()
            except Exception as e:
                self.log_ui_message(f"Indítás hiba: {e}", logging.ERROR); self.core = None
        asyncio.create_task(_run())

    @QtCore.Slot()
    def on_stop(self):
        if not self.core: return
        core = self.core; self.core = None
        self.log_ui_message("Kapcsolat leállítása kezdeményezve.")
        self._pointer_timer.stop()
        self.hide_pointer_overlay()
        asyncio.create_task(core.stop())

    @QtCore.Slot()
    def on_ping(self):
        if not self.core: return
        self.log_ui_message("Ping üzenet küldése.")
        asyncio.create_task(self.core.send_ping())

    @QtCore.Slot()
    def on_bw(self):
        if not self.core: return
        self.log_ui_message("Sávszélesség teszt indítása.")
        asyncio.create_task(self.core.run_bw_test(BW_SECONDS, BW_CHUNK))

    @QtCore.Slot()
    def on_share_start(self):
        if not self.core:
            self.log_ui_message("Előbb indítsd el a kapcsolatot.", logging.WARNING); return
        label = self.res_combo.currentText()
        width, height = RESOLUTIONS[label]
        fps = self.fps_slider.value(); kbps = self.br_slider.value()
        share_audio = self.chk_share_audio.isChecked()
        self.log_ui_message(
            f"Képernyőmegosztás indítása: {label}, {fps} FPS, {kbps} kbps"
            + (" + hang" if share_audio else "")
        )
        asyncio.create_task(self.core.start_share(width, height, fps, kbps, share_audio))

    @QtCore.Slot()
    def on_share_stop(self):
        if self.core:
            self.log_ui_message("Képernyőmegosztás leállítása.")
            self.hide_pointer_overlay()
            asyncio.create_task(self.core.stop_share())

    @QtCore.Slot(int)
    def on_role_changed(self, index: int) -> None:
        role_label = self.role_combo.itemText(index)
        role_value = self.role_combo.itemData(index) or "sender"
        self._save_setting("ui/role", role_label)
        self._update_role_ui(role_value)

    @QtCore.Slot()
    def on_res_changed(self):
        if not self.core: return
        label = self.res_combo.currentText()
        width, height = RESOLUTIONS[label]
        self.log_ui_message(f"Felbontás váltása: {label}")
        asyncio.create_task(self.core.set_resolution(width, height))

    @QtCore.Slot()
    def on_fps_changed(self, val: int):
        self.fps_label.setText(f"FPS: {val}")
        self._save_setting("ui/fps", val)
        self._pending_fps = val
        self._fps_debounce.start(200)

    @QtCore.Slot()
    def on_br_changed(self, val: int):
        self.br_label.setText(f"Bitráta: {val} kbps")
        self._save_setting("ui/bitrate", val)
        self._pending_bitrate = val
        self._br_debounce.start(200)

    @QtCore.Slot(int)
    def on_audio_volume_changed(self, value: int):
        self.audio_volume_label.setText(f"Hang: {value}%")
        self._save_setting("ui/audio_volume", value)
        if self.core:
            self.core.set_audio_volume(value / 100.0)

    @QtCore.Slot(QtGui.QImage)
    def update_video(self, img: QtGui.QImage):
        if img.isNull(): return
        pix = QtGui.QPixmap.fromImage(img)
        self._last_pixmap = pix
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.update_pixmap(pix)
            if self.pointer_overlay.isVisible() and self._pointer_norm is not None:
                nx, ny = self._pointer_norm
                self.fullscreen_window.update_pointer(nx, ny, True)
            else:
                self.fullscreen_window.hide_pointer_overlay()
        scaled = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self._reposition_pointer_overlay()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        if self._last_pixmap:
            scaled = self._last_pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)
        self._reposition_pointer_overlay()
        return super().resizeEvent(e)

    def _prepare_cursor_pixmap(self) -> Optional[QtGui.QPixmap]:
        if self._cursor_pixmap.isNull():
            return None
        pixmap = self.video_label.pixmap()
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

    def _show_pointer_overlay(self, norm_x: float, norm_y: float, *, local: bool) -> None:
        pix = self._prepare_cursor_pixmap()
        if pix is None:
            return
        nx = max(0.0, min(1.0, float(norm_x)))
        ny = max(0.0, min(1.0, float(norm_y)))
        self.pointer_overlay.setPixmap(pix)
        self.pointer_overlay.resize(pix.size())
        self._pointer_norm = (nx, ny)
        self._pointer_local = local
        self.pointer_overlay.show()
        self._reposition_pointer_overlay()
        self._pointer_timer.start(2500)
        if self.fullscreen_window:
            self.fullscreen_window.update_pointer(nx, ny, True)

    def _reposition_pointer_overlay(self) -> None:
        if self._pointer_norm is None or not self.pointer_overlay.isVisible():
            return
        pixmap = self.video_label.pixmap()
        if not pixmap or pixmap.isNull():
            self.hide_pointer_overlay()
            return
        lw, lh = self.video_label.width(), self.video_label.height()
        pw, ph = pixmap.width(), pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        offset_x = max(0, (lw - pw) // 2)
        offset_y = max(0, (lh - ph) // 2)
        nx, ny = self._pointer_norm
        x = offset_x + int(round(nx * pw))
        y = offset_y + int(round(ny * ph))
        ow = self.pointer_overlay.width()
        oh = self.pointer_overlay.height()
        x = max(0, min(x, lw - ow))
        y = max(0, min(y, lh - oh))
        self.pointer_overlay.move(x, y)
        if self.fullscreen_window and self.fullscreen_window.isVisible() and self._pointer_norm is not None:
            nx, ny = self._pointer_norm
            self.fullscreen_window.update_pointer(nx, ny, True)

    def hide_pointer_overlay(self) -> None:
        self._pointer_timer.stop()
        self._pointer_norm = None
        self._pointer_local = False
        self.pointer_overlay.hide()
        if self.fullscreen_window:
            self.fullscreen_window.hide_pointer_overlay()
        self._screen_pointer_overlay.hide_pointer()

    @QtCore.Slot(float, float, bool)
    def on_pointer_update(self, norm_x: float, norm_y: float, visible: bool) -> None:
        if visible:
            self._show_pointer_overlay(norm_x, norm_y, local=False)
            if self.core and self.core.role == "sender":
                bbox = self.core.get_capture_geometry()
                if bbox:
                    self._screen_pointer_overlay.show_pointer(norm_x, norm_y, bbox)
                else:
                    self._screen_pointer_overlay.hide_pointer()
        else:
            self.hide_pointer_overlay()

    @QtCore.Slot(float, float)
    def on_video_clicked(self, norm_x: float, norm_y: float) -> None:
        if not self.core or self.core.role != "receiver":
            return
        pixmap = self.video_label.pixmap()
        if not pixmap or pixmap.isNull():
            return
        self._show_pointer_overlay(norm_x, norm_y, local=True)
        self.log_ui_message("Kurzor mutatása elküldve a küldőnek.")
        asyncio.create_task(self.core.send_pointer(norm_x, norm_y))

    @QtCore.Slot(float, float)
    def on_fullscreen_clicked(self, norm_x: float, norm_y: float) -> None:
        if not self.core or self.core.role != "receiver":
            return
        pixmap = self.video_label.pixmap()
        if not pixmap or pixmap.isNull():
            return
        self._show_pointer_overlay(norm_x, norm_y, local=True)
        self.log_ui_message("Kurzor mutatása elküldve a küldőnek.")
        asyncio.create_task(self.core.send_pointer(norm_x, norm_y))

    @QtCore.Slot()
    def on_pointer_timeout(self) -> None:
        was_local = self._pointer_local
        self.hide_pointer_overlay()
        if was_local and self.core and self.core.role == "receiver":
            asyncio.create_task(self.core.send_pointer_hide())

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() == QtCore.QEvent.WindowStateChange and self._tray_icon and self._tray_icon.isVisible():
            if self.isMinimized():
                QtCore.QTimer.singleShot(0, self.hide)
                if not self._tray_message_shown:
                    self._tray_icon.showMessage(
                        "VatiVision Pro",
                        "Az alkalmazás tovább fut a tálcán.",
                        QtWidgets.QSystemTrayIcon.Information,
                        3000,
                    )
                    self._tray_message_shown = True
        super().changeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.close()
        self._save_setting("window/geometry", self.saveGeometry())
        self.settings.sync()
        self._screen_pointer_overlay.close()
        if self._tray_icon and self._tray_icon.isVisible() and not self._allow_close:
            event.ignore()
            self.hide()
            if not self._tray_message_shown:
                self._tray_icon.showMessage(
                    "VatiVision Pro",
                    "Az alkalmazás tovább fut a tálcán.",
                    QtWidgets.QSystemTrayIcon.Information,
                    3000,
                )
                self._tray_message_shown = True
            return
        super().closeEvent(event)

    def _create_tray_icon(self) -> None:
        self._tray_icon = QtWidgets.QSystemTrayIcon(self._app_icon, self)
        self._tray_icon.setToolTip(APP_TITLE)

        menu = QtWidgets.QMenu()
        show_action = menu.addAction("Megnyitás")
        show_action.triggered.connect(self._restore_from_tray)
        menu.addSeparator()
        quit_action = menu.addAction("Kilépés")
        quit_action.triggered.connect(self._quit_app)

        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    def _restore_from_tray(self) -> None:
        if not self._tray_icon:
            return
        self.showNormal()
        self.activateWindow()
        self.raise_()
        self._tray_message_shown = True

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QtWidgets.QSystemTrayIcon.Trigger,
            QtWidgets.QSystemTrayIcon.DoubleClick,
        ):
            self._restore_from_tray()

    def _quit_app(self) -> None:
        self._allow_close = True
        if self._tray_icon:
            self._tray_icon.hide()
        if self.core:
            self.on_stop()
        self.close()

    @QtCore.Slot()
    def on_toggle_fullscreen(self) -> None:
        if self.fullscreen_window is None:
            self.fullscreen_window = FullscreenViewer(self)
            self.fullscreen_window.closed.connect(self.on_fullscreen_closed)
            self.fullscreen_window.clicked.connect(self.on_fullscreen_clicked)
            self.fullscreen_window.set_cursor_source(self._cursor_pixmap)

        if self.fullscreen_window.isVisible():
            self.fullscreen_window.close()
            self.btn_fullscreen.setText("Teljes képernyő")
            return

        if self._last_pixmap:
            self.fullscreen_window.update_pixmap(self._last_pixmap)
        if self.pointer_overlay.isVisible() and self._pointer_norm is not None:
            nx, ny = self._pointer_norm
            self.fullscreen_window.update_pointer(nx, ny, True)
        else:
            self.fullscreen_window.hide_pointer_overlay()
        self.fullscreen_window.showFullScreen()
        self.fullscreen_window.activateWindow()
        self.btn_fullscreen.setText("Kilépés a teljes képernyőből")

    @QtCore.Slot()
    def on_fullscreen_closed(self) -> None:
        self.btn_fullscreen.setText("Teljes képernyő")

    def _restore_settings(self) -> None:
        geometry = self.settings.value("window/geometry")
        if isinstance(geometry, (QtCore.QByteArray, bytes)) and geometry:
            self.restoreGeometry(QtCore.QByteArray(geometry))

        role = self.settings.value("ui/role", "sender")
        idx = self.role_combo.findText(str(role))
        if idx >= 0:
            self.role_combo.setCurrentIndex(idx)

        prefer = self.settings.value("ui/prefer_relay", False)
        self.chk_relay.setChecked(self._to_bool(prefer))

        share_audio = self.settings.value("ui/share_audio", False)
        self.chk_share_audio.setChecked(self._to_bool(share_audio))

        res = self.settings.value("ui/resolution")
        if res:
            r_idx = self.res_combo.findText(str(res))
            if r_idx >= 0:
                self.res_combo.setCurrentIndex(r_idx)

        fps = self._to_int(
            self.settings.value("ui/fps", self.fps_slider.value()),
            fallback=self.fps_slider.value(),
        )
        self.fps_slider.setValue(fps)

        bitrate = self._to_int(
            self.settings.value("ui/bitrate", self.br_slider.value()),
            fallback=self.br_slider.value(),
        )
        self.br_slider.setValue(bitrate)

        audio_vol = self._to_int(
            self.settings.value("ui/audio_volume", self.audio_volume_slider.value()),
            fallback=self.audio_volume_slider.value(),
        )
        self.audio_volume_slider.setValue(audio_vol)
        self.audio_volume_label.setText(f"Hang: {audio_vol}%")

    def _save_setting(self, key: str, value) -> None:
        self.settings.setValue(key, value)

    def _update_role_ui(self, role_value: str) -> None:
        is_sender = role_value == "sender"
        self.share_group.setVisible(is_sender)
        show_audio = role_value == "receiver"
        self.audio_volume_label.setVisible(show_audio)
        self.audio_volume_slider.setVisible(show_audio)
        if not is_sender:
            self._screen_pointer_overlay.hide_pointer()

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "t", "yes", "on"}
        return bool(value)

    @staticmethod
    def _to_int(value, fallback: Optional[int] = None) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            if fallback is not None:
                return fallback
            raise

def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(style())
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QtGui.QIcon(str(APP_ICON_PATH)))
    loop = QEventLoop(app); asyncio.set_event_loop(loop)
    w = Main(); w.show()
    with loop:
        loop.run_forever()
