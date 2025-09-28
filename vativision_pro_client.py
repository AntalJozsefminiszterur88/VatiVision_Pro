
# VatiVision Pro — Client (Discord-style UI) — fixed import name
import asyncio, json
import logging
from logging.handlers import RotatingFileHandler
from time import perf_counter
from typing import Optional
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
from qasync import QEventLoop
from aiohttp import ClientSession, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCDataChannel

from vati_screenshare import ScreenShareTrack, RESOLUTIONS

LOG_ROOT = Path.home() / "felhasználó" / "dokumentumok" / "UMKGL Solutions" / "VatiVision_Pro"


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
APP_TITLE    = "VatiVision Pro — Kliens (Ping + TURN + Sávszél + Képernyő)"
APP_ICON_PATH = Path(__file__).resolve().parent / "program_logo.png"

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
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await wait_ice_complete(self.pc)
            await self.ws.send_json({"op":"offer","sdp":{
                "type": self.pc.localDescription.type,
                "sdp":  self.pc.localDescription.sdp,
            }})

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
            self.msg_in.emit(str(m)); return

        if isinstance(m, (bytes, bytearray)) and self._bw_recv_active:
            self._bw_recv_bytes += len(m)

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
                    sdp = data["sdp"]
                    await self.pc.setRemoteDescription(RTCSessionDescription(**sdp))
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

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await wait_ice_complete(self.pc)
        await self.ws.send_json({"op":"offer","sdp":{
            "type": self.pc.localDescription.type,
            "sdp":  self.pc.localDescription.sdp,
        }})
        self._emit_log("[media] Képernyőmegosztás elindítva.")

    async def stop_share(self):
        if not self.pc: return
        if self._video_sender:
            try: self.pc.removeTrack(self._video_sender)
            except Exception: pass
            self._video_sender = None
        if self._share_track:
            try: await self._share_track.stop()
            except Exception: pass
            self._share_track = None
        try:
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await wait_ice_complete(self.pc)
            await self.ws.send_json({"op":"offer","sdp":{
                "type": self.pc.localDescription.type,
                "sdp":  self.pc.localDescription.sdp,
            }})
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

    async def stop(self):
        try:
            if self.ws: await self.ws.close()
        except: pass
        try:
            if self.session: await self.session.close()
        except: pass
        try:
            if self.pc: await self.pc.close()
        except: pass
        self.status.emit("Leállítva")

class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 720)
        if APP_ICON_PATH.exists():
            icon = QtGui.QIcon(str(APP_ICON_PATH))
            self.setWindowIcon(icon)

        self.ui_logger = logging.getLogger(f"{__name__}.UI")

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(12,12,12,12); root.setSpacing(12)

        side = QtWidgets.QFrame(); side.setFixedWidth(340)
        side.setStyleSheet(f"QFrame {{ background-color:{DARK_CARD}; border-radius:12px; }}")
        s = QtWidgets.QVBoxLayout(side); s.setContentsMargins(12,12,12,12); s.setSpacing(10)

        head = QtWidgets.QLabel(f"<b>VatiVision Pro</b><br>"
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
        self.video_label = QtWidgets.QLabel("Nincs bejövő videó")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumHeight(280)
        self.video_label.setStyleSheet("QLabel { background-color: #1e1f22; border: 1px solid #3a3c41; border-radius: 8px; }")
        pv.addWidget(self.video_label)
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

        self.role_combo.currentTextChanged.connect(
            lambda value: self._save_setting("ui/role", value)
        )
        self.chk_relay.toggled.connect(
            lambda checked: self._save_setting("ui/prefer_relay", checked)
        )
        self.res_combo.currentTextChanged.connect(
            lambda value: self._save_setting("ui/resolution", value)
        )

        self._restore_settings()

    @QtCore.Slot(str)
    def append_log_message(self, message: str) -> None:
        self.log.appendPlainText(message)

    def log_ui_message(self, message: str, level: int = logging.INFO) -> None:
        self.ui_logger.log(level, message)
        self.log.appendPlainText(message)

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
        if self.core:
            self.log_ui_message(f"FPS módosítása: {val}")
            asyncio.create_task(self.core.set_fps(val))

    @QtCore.Slot()
    def on_br_changed(self, val: int):
        self.br_label.setText(f"Bitráta: {val} kbps")
        self._save_setting("ui/bitrate", val)
        if self.core:
            self.log_ui_message(f"Bitráta módosítása: {val} kbps")
            asyncio.create_task(self.core.set_bitrate(val))

    @QtCore.Slot(QtGui.QImage)
    def update_video(self, img: QtGui.QImage):
        if img.isNull(): return
        pix = QtGui.QPixmap.fromImage(img)
        scaled = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        if self.video_label.pixmap() is not None:
            pix = self.video_label.pixmap()
            if pix:
                scaled = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled)
        return super().resizeEvent(e)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_setting("window/geometry", self.saveGeometry())
        self.settings.sync()
        super().closeEvent(event)

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
