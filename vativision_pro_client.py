
# VatiVision Pro — Client (Discord-style UI) — fixed import name
import asyncio, json
import logging
import os
import sys
import ctypes
from logging.handlers import RotatingFileHandler
from time import perf_counter
from typing import Optional, Tuple
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
from qasync import QEventLoop
from aiohttp import ClientSession, WSMsgType
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel,
)
from aiortc.exceptions import InvalidStateError

from vati_screenshare import ScreenShareTrack, RESOLUTIONS

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


LOG_ROOT = _resolve_documents_dir() / "UMKGL Solutions" / "VatiVision_Pro"


def _setup_file_logging() -> Path:
    """Initialise rotating file logging limited to 5 MB in the requested folder."""

    log_dir = LOG_ROOT
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


LOG_FILE = _setup_file_logging()
logger = logging.getLogger(__name__)
logger.info("Fájllog inicializálva: %s", LOG_FILE)

SIGNALING_WS = "wss://vatib-vezerlo.duckdns.org/ws"
TURN_HOST    = "vatib-vezerlo.duckdns.org"
ROOM_NAME    = "VATI-ROOM"
ROOM_PIN     = "428913"
STUN_URLS    = ["stun:stun.l.google.com:19302"]
APP_TITLE    = "VatiVision Pro - UMKGL Solutions"
APP_ICON_PATH = Path(__file__).resolve().parent / "program_logo.png"
CURSOR_PATH   = Path(__file__).resolve().parent / "cursor.png"

BW_SECONDS   = 5.0
BW_CHUNK     = 16 * 1024
BW_MAX_BUF   = 4 * 1024 * 1024

DARK_BG = "#2b2d31"; DARK_CARD = "#313338"; DARK_ELEV = "#1e1f22"; ACCENT = "#5865f2"; TEXT="#e3e5e8"
def style():
    return f"""
    QWidget {{ background-color:{DARK_BG}; color:{TEXT}; font-family:Segoe UI, Inter, Roboto, Arial; font-size:14px; }}
    QGroupBox {{ background-color:{DARK_CARD}; border:1px solid #3a3c41; border-radius:10px; margin-top:16px; padding:12px; }}
    QPushButton {{ background-color:{ACCENT}; color:white; border:none; border-radius:8px; padding:8px 12px; font-weight:600; }}
    QLabel#status {{ background-color:{DARK_CARD}; border:1px solid #3a3c41; border-radius:8px; padding:8px; }}
    QPlainTextEdit {{ background-color:{DARK_ELEV}; border:1px solid #3a3c41; border-radius:8px; }}
    QComboBox, QLineEdit, QSlider {{ background-color:{DARK_ELEV}; border:1px solid #3a3c41; border-radius:8px; padding:6px; }}
    QCheckBox {{ background:transparent; }}
    """


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


class PointerCanvas(QtWidgets.QWidget):
    """Widget that renders video frames with an optional overlay cursor."""

    pointerClicked = QtCore.Signal(float, float)
    pointerCleared = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, interactive: bool = True):
        super().__init__(parent)
        self._interactive = interactive
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._target_rect = QtCore.QRect()
        self._pointer_visible = False
        self._pointer_pos = QtCore.QPointF(0.0, 0.0)
        self._cursor_pixmap = QtGui.QPixmap(str(CURSOR_PATH)) if CURSOR_PATH.exists() else QtGui.QPixmap()
        self._placeholder = "Nincs bejövő videó"
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setMouseTracking(interactive)
        if interactive:
            self.setCursor(QtCore.Qt.CrossCursor)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def show_pointer(self, norm_x: float, norm_y: float) -> None:
        self._pointer_visible = True
        self._pointer_pos = QtCore.QPointF(float(norm_x), float(norm_y))
        self.update()

    def clear_pointer(self) -> None:
        if not self._pointer_visible:
            return
        self._pointer_visible = False
        self.update()

    def has_pixmap(self) -> bool:
        return bool(self._pixmap and not self._pixmap.isNull())

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(640, 360)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1f22"))

        if not self.has_pixmap():
            painter.setPen(QtGui.QColor("#888888"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self._placeholder)
            self._target_rect = QtCore.QRect()
            painter.setPen(QtGui.QPen(QtGui.QColor("#3a3c41")))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
            return

        pix = self._pixmap
        assert pix is not None
        scaled = pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        target = QtCore.QRect(x, y, scaled.width(), scaled.height())
        self._target_rect = target
        painter.drawPixmap(target.topLeft(), scaled)

        if self._pointer_visible and not self._cursor_pixmap.isNull():
            cursor = self._cursor_pixmap
            cur_w = cursor.width()
            cur_h = cursor.height()
            if target.width() > 0 and target.height() > 0:
                nx = max(0.0, min(1.0, float(self._pointer_pos.x())))
                ny = max(0.0, min(1.0, float(self._pointer_pos.y())))
                px = target.left() + int(round(nx * max(0, target.width() - 1)))
                py = target.top() + int(round(ny * max(0, target.height() - 1)))
                px = max(target.left(), min(px, target.right() - cur_w + 1))
                py = max(target.top(), min(py, target.bottom() - cur_h + 1))
                painter.drawPixmap(px, py, cursor)

        painter.setPen(QtGui.QPen(QtGui.QColor("#3a3c41")))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self._interactive:
            return super().mousePressEvent(event)

        if event.button() == QtCore.Qt.RightButton:
            self.pointerCleared.emit()
            event.accept()
            return

        if event.button() == QtCore.Qt.LeftButton and self.has_pixmap():
            if not self._target_rect.contains(event.position().toPoint()):
                return
            target = self._target_rect
            if target.width() <= 0 or target.height() <= 0:
                return
            rel_x = (event.position().x() - target.left()) / float(target.width())
            rel_y = (event.position().y() - target.top()) / float(target.height())
            rel_x = min(max(rel_x, 0.0), 1.0)
            rel_y = min(max(rel_y, 0.0), 1.0)
            self.pointerClicked.emit(rel_x, rel_y)
            event.accept()
            return

        super().mousePressEvent(event)


class FullscreenViewer(QtWidgets.QWidget):
    """Window that displays the shared video feed in fullscreen."""

    closed = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("VatiVision Pro — Teljes képernyő")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PointerCanvas(self, interactive=False)
        layout.addWidget(self.canvas, 1)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.canvas.set_pixmap(pixmap)

    def update_pointer(self, norm_x: float, norm_y: float) -> None:
        self.canvas.show_pointer(norm_x, norm_y)

    def clear_pointer(self) -> None:
        self.canvas.clear_pointer()


def make_rtc_config(prefer_relay: bool) -> RTCConfiguration:
    ice_servers = [
        RTCIceServer(urls=STUN_URLS),
        RTCIceServer(
            urls=[f"turn:{TURN_HOST}:3478?transport=udp", f"turn:{TURN_HOST}:3478?transport=tcp"],
            username="vati", credential="SuperSecret123",
        ),
    ]
    cfg = RTCConfiguration(iceServers=ice_servers)
    if prefer_relay:
        cfg.iceTransportPolicy = "relay"
    return cfg

async def wait_ice_complete(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        return
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(True)
    await fut

class Core(QtCore.QObject):
    status      = QtCore.Signal(str)
    log         = QtCore.Signal(str)
    msg_in      = QtCore.Signal(str)
    video_frame = QtCore.Signal(QtGui.QImage)
    pointer_update = QtCore.Signal(float, float)
    pointer_cleared = QtCore.Signal()

    def __init__(self, role: str, prefer_relay: bool):
        super().__init__()
        self.role = role
        self.prefer_relay = prefer_relay
        self.pc: Optional[RTCPeerConnection] = None
        self.session: Optional[ClientSession] = None
        self.ws = None
        self.channel: Optional[RTCDataChannel] = None

        self._logger = logging.getLogger(f"{__name__}.Core")

        self._bw_recv_active = False
        self._bw_recv_bytes  = 0
        self._bw_recv_t0     = 0.0
        self._bw_ready_fut: Optional[asyncio.Future] = None
        self._bw_report_fut: Optional[asyncio.Future] = None

        self._share_track: Optional[ScreenShareTrack] = None
        self._video_sender = None
        self._target_bitrate_bps: Optional[int] = None

        self._pending_answer: Optional[asyncio.Future] = None
        self._offer_lock = asyncio.Lock()
        self._waiting_for_answer_logged = False
        self._last_pointer: Optional[Tuple[float, float]] = None

    def _emit_log(self, message: str, level: int = logging.INFO) -> None:
        self._logger.log(level, message)
        self.log.emit(message)

    async def start(self):
        self.status.emit("Indítás...")
        self.pc = RTCPeerConnection(configuration=make_rtc_config(self.prefer_relay))

        @self.pc.on("iceconnectionstatechange")
        def _ice():
            self.status.emit(f"ICE: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "connected":
                self.status.emit("ICE: kapcsolódva ✅")

        @self.pc.on("datachannel")
        def _dc(ch: RTCDataChannel):
            self.channel = ch
            @ch.on("message")
            def _on(m):
                self._handle_message(m)

        @self.pc.on("track")
        def _on_track(track):
            if track.kind == "video":
                asyncio.create_task(self._consume_video(track))

        if self.role == "sender":
            self.channel = self.pc.createDataChannel("chat")
            @self.channel.on("message")
            def _on_msg(m):
                self._handle_message(m)

        self.session = ClientSession()
        self.ws = await self.session.ws_connect(SIGNALING_WS, heartbeat=30, autoping=True)
        await self.ws.send_json({"op":"hello","room":ROOM_NAME,"pin":ROOM_PIN,"role":self.role})
        hello = await self.ws.receive()
        if hello.type != WSMsgType.TEXT or json.loads(hello.data).get("op") != "hello_ok":
            raise RuntimeError("Signaling hello failed")

        if self.role == "sender":
            await self._create_and_send_offer()

        asyncio.create_task(self._ws_loop())
        self.status.emit("Várakozás a partnerre...")

    async def _consume_video(self, track):
        try:
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="rgb24")
                h, w, _ = img.shape
                qimg = QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format_RGB888).copy()
                self.video_frame.emit(qimg)
        except Exception as e:
            self._emit_log(f"[media] video consumer ended: {e}", logging.ERROR)

    def _handle_message(self, m):
        if isinstance(m, str):
            if m == "ping":
                self.msg_in.emit("Ping érkezett ✅")
                if self.channel and self.channel.readyState == "open":
                    try:
                        self.channel.send("pong")
                    except Exception as exc:
                        self._emit_log(f"Ping visszajelzés küldési hiba: {exc}", logging.WARNING)
                return
            if m == "pong":
                self.msg_in.emit("Pong visszaigazolás megérkezett ✅")
                return
            if m.startswith("BW_BEGIN"):
                self._bw_recv_active = True
                self._bw_recv_bytes  = 0
                self._bw_recv_t0     = perf_counter()
                try: self.channel.send("BW_READY")
                except Exception: pass
                return
            if m == "BW_END":
                if self._bw_recv_active:
                    elapsed = max(1e-6, perf_counter() - self._bw_recv_t0)
                    mbps = (self._bw_recv_bytes * 8) / 1_000_000 / elapsed
                    report = f"BW_REPORT {mbps:.2f} Mb/s ({self._bw_recv_bytes/1_000_000:.2f} MB in {elapsed:.2f} s)"
                    try: self.channel.send(report)
                    except Exception: pass
                    self.msg_in.emit(report)
                self._bw_recv_active = False
                return
            if m == "BW_READY":
                if self._bw_ready_fut and not self._bw_ready_fut.done():
                    self._bw_ready_fut.set_result(True)
                return
            if m.startswith("BW_REPORT"):
                self.msg_in.emit(m)
                if self._bw_report_fut and not self._bw_report_fut.done():
                    self._bw_report_fut.set_result(m)
                return
            if m.startswith("PTR "):
                parts = m.split()
                if len(parts) == 3:
                    try:
                        nx = float(parts[1])
                        ny = float(parts[2])
                    except ValueError:
                        self._emit_log(f"[pointer] Hibás koordináták: {m}", logging.WARNING)
                    else:
                        self._register_pointer_update(nx, ny)
                return
            if m == "PTR_CLEAR":
                self._register_pointer_clear()
                return
            self.msg_in.emit(str(m)); return

        if isinstance(m, (bytes, bytearray)) and self._bw_recv_active:
            self._bw_recv_bytes += len(m)

    async def _wait_for_pending_answer(self) -> None:
        fut = self._pending_answer
        if fut and not fut.done():
            if not self._waiting_for_answer_logged:
                self._emit_log(
                    "Várakozás a korábbi ajánlatra érkező válaszra a következő művelet előtt.",
                    logging.INFO,
                )
                self._waiting_for_answer_logged = True
            try:
                await asyncio.shield(fut)
            except Exception:
                pass
            finally:
                self._waiting_for_answer_logged = False

    def _clear_pending_answer(
        self,
        *,
        exc: Optional[BaseException] = None,
        result: Optional[bool] = True,
        cancel: bool = False,
    ) -> None:
        fut = self._pending_answer
        if fut and not fut.done():
            if cancel:
                fut.cancel()
            elif exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        self._pending_answer = None
        self._waiting_for_answer_logged = False

    async def _create_and_send_offer(self) -> None:
        if not self.pc:
            raise RuntimeError("Nincs aktív PeerConnection az ajánlat küldéséhez.")
        if not self.ws:
            raise RuntimeError("A jelzéscsatorna nem elérhető az ajánlat küldéséhez.")

        await self._wait_for_pending_answer()

        async with self._offer_lock:
            state = getattr(self.pc, "signalingState", None)
            if state and state != "stable":
                self._emit_log(
                    f"Az ajánlat készítése kihagyva – a jelzési állapot nem stabil: {state}.",
                    logging.WARNING,
                )
                return

            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await wait_ice_complete(self.pc)

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._pending_answer = fut

            try:
                await self.ws.send_json({
                    "op": "offer",
                    "sdp": {
                        "type": self.pc.localDescription.type,
                        "sdp": self.pc.localDescription.sdp,
                    },
                })
            except Exception as exc:
                self._clear_pending_answer(exc=exc)
                raise

    async def _ws_loop(self):
        try:
            async for m in self.ws:
                if m.type != WSMsgType.TEXT:
                    continue
                data = json.loads(m.data)
                op = data.get("op")
                if op == "offer" and self.role == "receiver":
                    sdp = data["sdp"]
                    await self.pc.setRemoteDescription(RTCSessionDescription(**sdp))
                    ans = await self.pc.createAnswer()
                    await self.pc.setLocalDescription(ans)
                    await wait_ice_complete(self.pc)
                    await self.ws.send_json({"op":"answer","sdp":{
                        "type": self.pc.localDescription.type,
                        "sdp":  self.pc.localDescription.sdp,
                    }})
                elif op == "answer" and self.role == "sender":
                    if not self.pc:
                        continue

                    sdp = data["sdp"]
                    state = self.pc.signalingState
                    if state != "have-local-offer":
                        self._emit_log(
                            f"Váratlan answer érkezett a(z) {state} jelzési állapotban – figyelmen kívül hagyva.",
                            logging.WARNING,
                        )
                        self._clear_pending_answer(result=False)
                        continue

                    try:
                        await self.pc.setRemoteDescription(RTCSessionDescription(**sdp))
                    except InvalidStateError as e:
                        new_state = getattr(self.pc, "signalingState", "ismeretlen")
                        self._emit_log(
                            f"Távoli SDP beállítása érvénytelen állapotban (\"{new_state}\"): {e}",
                            logging.WARNING,
                        )
                        if new_state == "stable":
                            self._clear_pending_answer(result=True)
                        else:
                            self._clear_pending_answer(exc=e)
                        continue
                    except Exception as e:
                        self._emit_log(f"Távoli SDP beállítása sikertelen: {e}", logging.ERROR)
                        self._clear_pending_answer(exc=e)
                        continue

                    self._clear_pending_answer(result=True)
        except Exception as e:
            self._emit_log(f"WS loop ended: {e}", logging.ERROR)

    async def send_ping(self):
        if self.channel and self.channel.readyState == "open":
            self.channel.send("ping")

    async def run_bw_test(self, seconds: float = BW_SECONDS, chunk_size: int = BW_CHUNK):
        if not self.channel or self.channel.readyState != "open":
            self._emit_log("DataChannel nincs nyitva.", logging.WARNING); return
        loop = asyncio.get_running_loop()
        self._bw_ready_fut = loop.create_future()
        try: self.channel.send(f"BW_BEGIN d={seconds}")
        except Exception as e: self._emit_log(f"[bw] begin hiba: {e}", logging.ERROR); return
        try: await asyncio.wait_for(self._bw_ready_fut, timeout=5.0)
        except asyncio.TimeoutError: self._emit_log("[bw] receiver timeout", logging.WARNING); return
        finally: self._bw_ready_fut = None

        payload = b"\\x00" * chunk_size
        t0 = perf_counter(); stop_at = t0 + seconds; sent_bytes = 0
        while perf_counter() < stop_at and self.channel.readyState == "open":
            if getattr(self.channel, "bufferedAmount", 0) > BW_MAX_BUF:
                await asyncio.sleep(0.005); continue
            try: self.channel.send(payload); sent_bytes += len(payload)
            except Exception: break
            await asyncio.sleep(0)

        self._bw_report_fut = loop.create_future()
        try: self.channel.send("BW_END")
        except Exception: pass
        try: await asyncio.wait_for(self._bw_report_fut, timeout=10.0)
        except asyncio.TimeoutError:
            elapsed = max(1e-6, perf_counter() - t0)
            mbps = (sent_bytes * 8) / 1_000_000 / elapsed
            self._emit_log(f"[bw] sender-est {mbps:.2f} Mb/s")
        finally: self._bw_report_fut = None

    async def start_share(self, width: int, height: int, fps: int, bitrate_kbps: int):
        if not self.pc: self._emit_log("Nincs aktív PeerConnection.", logging.WARNING); return
        if self._share_track is not None: self._emit_log("Már fut a megosztás.", logging.WARNING); return

        self._share_track = ScreenShareTrack(width=width, height=height, fps=fps)
        self._video_sender = self.pc.addTrack(self._share_track)

        if self._last_pointer and self.role == "sender":
            self._share_track.set_pointer(*self._last_pointer)

        self._target_bitrate_bps = max(50_000, int(bitrate_kbps) * 1000)
        try:
            get_params = getattr(self._video_sender, "getParameters", None)
            set_params = getattr(self._video_sender, "setParameters", None)
            if callable(get_params) and callable(set_params):
                params = get_params()
                if getattr(params, "encodings", None):
                    for enc in params.encodings:
                        enc.maxBitrate  = self._target_bitrate_bps
                        enc.maxFramerate = max(1, fps)
                await set_params(params)
            else:
                self._emit_log("[media] Az aiortc verziód nem támogatja a setParameters-t (bitráta/fps encoder-szinten).", logging.WARNING)
        except Exception as e:
            self._emit_log(f"[media] setParameters hiba (indítás): {e}", logging.ERROR)

        await self._create_and_send_offer()
        self._emit_log("[media] Képernyőmegosztás elindítva.")

    async def stop_share(self, *, renegotiate: bool = True):
        if not self.pc: return
        if self._video_sender:
            try: self.pc.removeTrack(self._video_sender)
            except Exception: pass
            self._video_sender = None
        if self._share_track:
            try: await self._share_track.stop()
            except Exception: pass
            self._share_track = None
        self._last_pointer = None
        self.pointer_cleared.emit()
        if renegotiate:
            try:
                await self._create_and_send_offer()
            except Exception as e:
                self._emit_log(f"[media] renegotiation hiba (stop_share): {e}", logging.ERROR)
        self._emit_log("[media] Képernyőmegosztás leállítva.")

    async def set_resolution(self, width: int, height: int):
        if self._share_track:
            self._share_track.set_size(width, height)
            self._emit_log(f"[media] Új felbontás: {width}×{height}")

    async def set_fps(self, fps: int):
        if self._share_track:
            self._share_track.set_fps(fps)
        if self._video_sender:
            try:
                get_params = getattr(self._video_sender, "getParameters", None)
                set_params = getattr(self._video_sender, "setParameters", None)
                if callable(get_params) and callable(set_params):
                    params = get_params()
                    if getattr(params, "encodings", None):
                        for enc in params.encodings:
                            enc.maxFramerate = max(1, fps)
                    await set_params(params)
                else:
                    self._emit_log("[media] Az aiortc verziód nem támogatja az FPS setParameters-t; a track FPS attól még változik.", logging.WARNING)
            except Exception as e:
                self._emit_log(f"[media] setParameters (fps) hiba: {e}", logging.ERROR)
        self._emit_log(f"[media] Új FPS: {fps}")

    async def set_bitrate(self, kbps: int):
        self._target_bitrate_bps = max(50_000, int(kbps) * 1000)
        if self._video_sender:
            try:
                get_params = getattr(self._video_sender, "getParameters", None)
                set_params = getattr(self._video_sender, "setParameters", None)
                if callable(get_params) and callable(set_params):
                    params = get_params()
                    if getattr(params, "encodings", None):
                        for enc in params.encodings:
                            enc.maxBitrate = self._target_bitrate_bps
                    await set_params(params)
                else:
                    self._emit_log("[media] Az aiortc verziód nem támogatja a bitráta setParameters-t.", logging.WARNING)
            except Exception as e:
                self._emit_log(f"[media] setParameters (bitrate) hiba: {e}", logging.ERROR)

    async def send_pointer(self, norm_x: float, norm_y: float) -> None:
        if not self.channel or self.channel.readyState != "open":
            self._emit_log("[pointer] DataChannel nincs nyitva.", logging.WARNING)
            return
        message = f"PTR {norm_x:.6f} {norm_y:.6f}"
        try:
            self.channel.send(message)
        except Exception as exc:
            self._emit_log(f"[pointer] küldési hiba: {exc}", logging.ERROR)
            return
        self._emit_log(f"[pointer] Koordináták küldve: ({norm_x:.3f}, {norm_y:.3f})")

    async def clear_remote_pointer(self) -> None:
        if not self.channel or self.channel.readyState != "open":
            return
        try:
            self.channel.send("PTR_CLEAR")
        except Exception as exc:
            self._emit_log(f"[pointer] törlés küldési hiba: {exc}", logging.ERROR)

    async def stop(self):
        try:
            await self.stop_share(renegotiate=False)
        except Exception:
            pass
        try:
            if self.ws: await self.ws.close()
        except: pass
        try:
            if self.session: await self.session.close()
        except: pass
        try:
            if self.pc: await self.pc.close()
        except: pass
        self._clear_pending_answer(cancel=True)
        self.status.emit("Leállítva")

    def _register_pointer_update(self, nx: float, ny: float) -> None:
        self._last_pointer = (nx, ny)
        self.pointer_update.emit(nx, ny)
        if self.role == "sender" and self._share_track:
            self._share_track.set_pointer(nx, ny)

    def _register_pointer_clear(self) -> None:
        self._last_pointer = None
        self.pointer_cleared.emit()
        if self.role == "sender" and self._share_track:
            self._share_track.clear_pointer()

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
        sh.addWidget(self.btn_share_start, 3, 1)
        sh.addWidget(self.btn_share_stop, 3, 2)
        v.addWidget(share)

        preview_box = QtWidgets.QGroupBox("Bejövő videó (fogadó)")
        pv = QtWidgets.QVBoxLayout(preview_box)

        video_container = QtWidgets.QWidget()
        vc_layout = QtWidgets.QVBoxLayout(video_container)
        vc_layout.setContentsMargins(0, 0, 0, 0)
        vc_layout.setSpacing(6)

        self.video_view = PointerCanvas(interactive=True)
        self.video_view.pointerClicked.connect(self.on_video_pointer_clicked)
        self.video_view.pointerCleared.connect(self.on_video_pointer_cleared)
        self.video_view.setMinimumHeight(280)
        self.video_view.setStyleSheet(
            "PointerCanvas { background-color: #1e1f22; border: 1px solid #3a3c41; border-radius: 8px; }"
        )
        vc_layout.addWidget(self.video_view, 1)

        controls_row = QtWidgets.QHBoxLayout()
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

        self._restore_settings()
        self._update_role_ui(self.role_combo.currentData() or "sender")
        self.fullscreen_window: Optional[FullscreenViewer] = None
        self._last_pixmap: Optional[QtGui.QPixmap] = None
        self._current_pointer: Optional[Tuple[float, float]] = None

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
        self.core.pointer_update.connect(self.on_pointer_update)
        self.core.pointer_cleared.connect(self.on_pointer_cleared)

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
        self._current_pointer = None
        self.video_view.clear_pointer()
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.clear_pointer()
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
        self.log_ui_message(
            f"Képernyőmegosztás indítása: {label}, {fps} FPS, {kbps} kbps"
        )
        asyncio.create_task(self.core.start_share(width, height, fps, kbps))

    @QtCore.Slot()
    def on_share_stop(self):
        if self.core:
            self.log_ui_message("Képernyőmegosztás leállítása.")
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

    @QtCore.Slot(float, float)
    def on_video_pointer_clicked(self, nx: float, ny: float) -> None:
        self._current_pointer = (nx, ny)
        self.video_view.show_pointer(nx, ny)
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.update_pointer(nx, ny)
        if self.core and self.core.role == "receiver":
            self.log_ui_message(
                f"Mutató pozíció küldése: x={nx:.2f}, y={ny:.2f}"
            )
            asyncio.create_task(self.core.send_pointer(nx, ny))

    @QtCore.Slot()
    def on_video_pointer_cleared(self) -> None:
        if self._current_pointer is None:
            return
        self._current_pointer = None
        self.video_view.clear_pointer()
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.clear_pointer()
        if self.core and self.core.role == "receiver":
            self.log_ui_message("Mutató törlése küldése")
            asyncio.create_task(self.core.clear_remote_pointer())

    @QtCore.Slot(float, float)
    def on_pointer_update(self, nx: float, ny: float) -> None:
        self._current_pointer = (nx, ny)
        self.video_view.show_pointer(nx, ny)
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.update_pointer(nx, ny)
        self.inbox.appendPlainText(f"Mutató érkezett: x={nx:.2f}, y={ny:.2f}")

    @QtCore.Slot()
    def on_pointer_cleared(self) -> None:
        self._current_pointer = None
        self.video_view.clear_pointer()
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.clear_pointer()
        self.inbox.appendPlainText("Mutató törlése érkezett")

    @QtCore.Slot(QtGui.QImage)
    def update_video(self, img: QtGui.QImage):
        if img.isNull(): return
        pix = QtGui.QPixmap.fromImage(img)
        self._last_pixmap = pix
        self.video_view.set_pixmap(pix)
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.fullscreen_window.update_pixmap(pix)
            if self._current_pointer:
                self.fullscreen_window.update_pointer(*self._current_pointer)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        if self._last_pixmap:
            self.video_view.set_pixmap(self._last_pixmap)
        return super().resizeEvent(e)

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

        if self.fullscreen_window.isVisible():
            self.fullscreen_window.close()
            self.btn_fullscreen.setText("Teljes képernyő")
            return

        if self._last_pixmap:
            self.fullscreen_window.update_pixmap(self._last_pixmap)
        if self._current_pointer:
            self.fullscreen_window.update_pointer(*self._current_pointer)
        else:
            self.fullscreen_window.clear_pointer()
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

    def _save_setting(self, key: str, value) -> None:
        self.settings.setValue(key, value)

    def _update_role_ui(self, role_value: str) -> None:
        is_sender = role_value == "sender"
        self.share_group.setVisible(is_sender)

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
