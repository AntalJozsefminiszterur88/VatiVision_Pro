
# VatiVision Pro — Client (Discord-style UI) — fixed import name
import asyncio, json
from time import perf_counter
from typing import Optional

from PySide6 import QtWidgets, QtCore, QtGui
from qasync import QEventLoop
from aiohttp import ClientSession, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCDataChannel

from vati_screenshare import ScreenShareTrack, RESOLUTIONS

SIGNALING_WS = "wss://vatib-vezerlo.duckdns.org/ws"
TURN_HOST    = "vatib-vezerlo.duckdns.org"
ROOM_NAME    = "VATI-ROOM"
ROOM_PIN     = "428913"
STUN_URLS    = ["stun:stun.l.google.com:19302"]
APP_TITLE    = "VatiVision Pro — Kliens (Ping + TURN + Sávszél + Képernyő)"

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

        self._bw_recv_active = False
        self._bw_recv_bytes  = 0
        self._bw_recv_t0     = 0.0
        self._bw_ready_fut: Optional[asyncio.Future] = None
        self._bw_report_fut: Optional[asyncio.Future] = None

        self._share_track: Optional[ScreenShareTrack] = None
        self._video_sender = None
        self._target_bitrate_bps: Optional[int] = None

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
            self.log.emit(f"[media] video consumer ended: {e}")

    def _handle_message(self, m):
        if isinstance(m, str):
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
            self.log.emit(f"WS loop ended: {e}")

    async def send_ping(self):
        if self.channel and self.channel.readyState == "open":
            self.channel.send("ping")

    async def run_bw_test(self, seconds: float = BW_SECONDS, chunk_size: int = BW_CHUNK):
        if not self.channel or self.channel.readyState != "open":
            self.log.emit("DataChannel nincs nyitva."); return
        loop = asyncio.get_running_loop()
        self._bw_ready_fut = loop.create_future()
        try: self.channel.send(f"BW_BEGIN d={seconds}")
        except Exception as e: self.log.emit(f"[bw] begin hiba: {e}"); return
        try: await asyncio.wait_for(self._bw_ready_fut, timeout=5.0)
        except asyncio.TimeoutError: self.log.emit("[bw] receiver timeout"); return
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
            self.log.emit(f"[bw] sender-est {mbps:.2f} Mb/s")
        finally: self._bw_report_fut = None

    async def start_share(self, width: int, height: int, fps: int, bitrate_kbps: int):
        if not self.pc: self.log.emit("Nincs aktív PeerConnection."); return
        if self._share_track is not None: self.log.emit("Már fut a megosztás."); return

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
                self.log.emit("[media] Az aiortc verziód nem támogatja a setParameters-t (bitráta/fps encoder-szinten).")
        except Exception as e:
            self.log.emit(f"[media] setParameters hiba (indítás): {e}")

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await wait_ice_complete(self.pc)
        await self.ws.send_json({"op":"offer","sdp":{
            "type": self.pc.localDescription.type,
            "sdp":  self.pc.localDescription.sdp,
        }})
        self.log.emit("[media] Képernyőmegosztás elindítva.")

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
            self.log.emit(f"[media] renegotiation hiba (stop_share): {e}")
        self.log.emit("[media] Képernyőmegosztás leállítva.")

    async def set_resolution(self, width: int, height: int):
        if self._share_track:
            self._share_track.set_size(width, height)
            self.log.emit(f"[media] Új felbontás: {width}×{height}")

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
                    self.log.emit("[media] Az aiortc verziód nem támogatja az FPS setParameters-t; a track FPS attól még változik.")
            except Exception as e:
                self.log.emit(f"[media] setParameters (fps) hiba: {e}")
        self.log.emit(f"[media] Új FPS: {fps}")

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
                    self.log.emit("[media] Az aiortc verziód nem támogatja a bitráta setParameters-t.")
            except Exception as e:
                self.log.emit(f"[media] setParameters (bitrate) hiba: {e}")

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
        self.role_combo = QtWidgets.QComboBox(); self.role_combo.addItems(["sender","receiver"])
        row.addWidget(QtWidgets.QLabel("Szerep:")); row.addWidget(self.role_combo); row.addStretch(1)
        v.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Kapcsolat indítása")
        self.btn_stop  = QtWidgets.QPushButton("Leállítás")
        self.btn_ping  = QtWidgets.QPushButton("Ping küldése (Küldő)")
        self.btn_bw    = QtWidgets.QPushButton("Sávszél-teszt")
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

        self.btn_share_start = QtWidgets.QPushButton("Megosztás indítása")
        self.btn_share_stop  = QtWidgets.QPushButton("Megosztás leállítása")

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

    @QtCore.Slot()
    def on_start(self):
        if self.core: return
        role = self.role_combo.currentText(); prefer = self.chk_relay.isChecked()
        self.core = Core(role=role, prefer_relay=prefer)
        self.core.status.connect(self.status.setText)
        self.core.log.connect(self.log.appendPlainText)
        self.core.msg_in.connect(self.inbox.appendPlainText)
        self.core.video_frame.connect(self.update_video)

        async def _run():
            try: await self.core.start()
            except Exception as e:
                self.log.appendPlainText(f"Indítás hiba: {e}"); self.core = None
        asyncio.create_task(_run())

    @QtCore.Slot()
    def on_stop(self):
        if not self.core: return
        core = self.core; self.core = None
        asyncio.create_task(core.stop())

    @QtCore.Slot()
    def on_ping(self):
        if not self.core: return
        asyncio.create_task(self.core.send_ping())

    @QtCore.Slot()
    def on_bw(self):
        if not self.core: return
        asyncio.create_task(self.core.run_bw_test(BW_SECONDS, BW_CHUNK))

    @QtCore.Slot()
    def on_share_start(self):
        if not self.core:
            self.log.appendPlainText("Előbb indítsd el a kapcsolatot."); return
        label = self.res_combo.currentText()
        width, height = RESOLUTIONS[label]
        fps = self.fps_slider.value(); kbps = self.br_slider.value()
        asyncio.create_task(self.core.start_share(width, height, fps, kbps))

    @QtCore.Slot()
    def on_share_stop(self):
        if self.core: asyncio.create_task(self.core.stop_share())

    @QtCore.Slot()
    def on_res_changed(self):
        if not self.core: return
        label = self.res_combo.currentText()
        width, height = RESOLUTIONS[label]
        asyncio.create_task(self.core.set_resolution(width, height))

    @QtCore.Slot()
    def on_fps_changed(self, val: int):
        self.fps_label.setText(f"FPS: {val}")
        if self.core: asyncio.create_task(self.core.set_fps(val))

    @QtCore.Slot()
    def on_br_changed(self, val: int):
        self.br_label.setText(f"Bitráta: {val} kbps")
        if self.core: asyncio.create_task(self.core.set_bitrate(val))

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

def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(style())
    loop = QEventLoop(app); asyncio.set_event_loop(loop)
    w = Main(); w.show()
    with loop:
        loop.run_forever()
