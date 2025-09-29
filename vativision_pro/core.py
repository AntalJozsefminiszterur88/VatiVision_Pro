"""Core application logic for VatiVision Pro."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter
from typing import Optional, Tuple

from PySide6 import QtCore, QtGui
from aiohttp import ClientSession, WSMsgType
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel,
)
from aiortc.exceptions import InvalidStateError

from .config import (
    SIGNALING_WS,
    TURN_HOST,
    ROOM_NAME,
    ROOM_PIN,
    STUN_URLS,
    BW_SECONDS,
    BW_CHUNK,
    BW_MAX_BUF,
)
from .media.screenshare import ScreenShareTrack
from .media.audio import (
    SystemAudioTrack,
    AudioPlayback,
    audio_capture_supported,
    audio_playback_supported,
)

@dataclass
class ShareParams:
    width: int
    height: int
    fps: int
    bitrate_kbps: int
    share_audio: bool


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()

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
    pointer     = QtCore.Signal(float, float, bool)

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
        self._audio_track: Optional[SystemAudioTrack] = None
        self._audio_sender = None
        self._target_bitrate_bps: Optional[int] = None
        self._share_resume_params: Optional[ShareParams] = None
        self._audio_sink: Optional[AudioPlayback] = None
        self._audio_volume: float = 1.0

        self._pending_answer: Optional[asyncio.Future] = None
        self._offer_lock = asyncio.Lock()
        self._waiting_for_answer_logged = False

        self._stopping = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._restart_lock = asyncio.Lock()
        self._ws_task: Optional[asyncio.Task] = None
        self._pointer_hide_task: Optional[asyncio.Task] = None
        self._ice_retry_task: Optional[asyncio.Task] = None
        self._max_ice_restart_attempts: int = 5

        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempt = 0
        self._backoff_steps = (2.0, 4.0, 8.0, 15.0)
        self._max_backoff = 30.0

    def _emit_log(self, message: str, level: int = logging.INFO) -> None:
        self._logger.log(level, message)
        self.log.emit(message)

    def _set_state(self, state: ConnectionState) -> None:
        if self._state is state:
            return
        self._logger.debug("Állapotváltás: %s -> %s", self._state.name, state.name)
        self._state = state

    async def start(self):
        if self._state not in {ConnectionState.DISCONNECTED}:
            self._emit_log(
                f"Start kihagyva – a jelenlegi állapot: {self._state.name}.",
                logging.DEBUG,
            )
            return
        self._stopping = False
        self._reconnect_attempt = 0
        self._cancel_scheduled_reconnect()
        await self._attempt_initial_connection()

    async def _attempt_initial_connection(self) -> None:
        if self._stopping:
            return
        self._set_state(ConnectionState.CONNECTING)
        try:
            await self._establish_connection()
        except asyncio.CancelledError:
            self._set_state(ConnectionState.DISCONNECTED)
            raise
        except Exception as exc:
            if self._stopping:
                self._set_state(ConnectionState.DISCONNECTED)
                return
            reason = f"Sikertelen kapcsolódás: {exc}"
            self._emit_log(reason, logging.ERROR)
            self._schedule_reconnect(reason)
        else:
            self._mark_connected()

    def _mark_connected(self) -> None:
        self._reconnect_attempt = 0
        self._set_state(ConnectionState.CONNECTED)

    async def _establish_connection(self) -> None:
        self.status.emit("Indítás...")
        self.pc = RTCPeerConnection(configuration=make_rtc_config(self.prefer_relay))

        @self.pc.on("iceconnectionstatechange")
        def _ice():
            state = self.pc.iceConnectionState
            self.status.emit(f"ICE: {state}")
            if state == "connected":
                self.status.emit("ICE: kapcsolódva ✅")
                self._cancel_scheduled_reconnect()
                self._cancel_ice_retry()
                self._mark_connected()
            elif state == "checking":
                # Ha az ICE folyamat elakad az "ellenőrzés" állapotban, akkor
                # ütemezzünk egy késleltetett újrapróbálkozást. Így nem kell a
                # felhasználónak manuálisan újraindítania az alkalmazást.
                self._ensure_ice_retry(initial_delay=5.0)
            elif state in {"failed", "disconnected"}:
                self._emit_log(
                    f"ICE állapot: {state}. Újracsatlakozás indítása...",
                    logging.WARNING,
                )
                min_delay = 1.0 if state == "failed" else 5.0
                self._schedule_reconnect(f"ICE állapot: {state}.", min_delay=min_delay)
                if state == "failed":
                    self._ensure_ice_retry(force_restart=True)

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
            elif track.kind == "audio":
                asyncio.create_task(self._consume_audio(track))

        if self.role == "sender":
            self.channel = self.pc.createDataChannel("chat")

            @self.channel.on("message")
            def _on_msg(m):
                self._handle_message(m)

        try:
            self.session = ClientSession()
            self.ws = await self.session.ws_connect(SIGNALING_WS, heartbeat=30, autoping=True)
            await self.ws.send_json({"op":"hello","room":ROOM_NAME,"pin":ROOM_PIN,"role":self.role})
            hello = await self.ws.receive()
            if hello.type != WSMsgType.TEXT or json.loads(hello.data).get("op") != "hello_ok":
                raise RuntimeError("Signaling hello failed")

            if self.role == "sender":
                await self._create_and_send_offer()
        except Exception:
            await self._cleanup_connection(for_reconnect=True)
            raise

        ws = self.ws
        if ws is not None:
            self._ws_task = asyncio.create_task(self._ws_loop(ws))
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

    async def _consume_audio(self, track):
        sink = self._ensure_audio_sink()
        if sink is None:
            self._emit_log(
                "[media] Audio track érkezett, de a lejátszás nem támogatott ezen a rendszeren.",
                logging.INFO,
            )
            try:
                while True:
                    await track.recv()
            except Exception:
                return
        else:
            try:
                while True:
                    frame = await track.recv()
                    data = frame.to_ndarray(format="flt")
                    sink.submit(data)
            except Exception as e:
                self._emit_log(f"[media] audio consumer ended: {e}", logging.ERROR)

    def _ensure_audio_sink(self) -> Optional[AudioPlayback]:
        if not audio_playback_supported():
            return None
        if self._audio_sink is None:
            try:
                self._audio_sink = AudioPlayback(volume=self._audio_volume)
            except Exception as exc:
                self._emit_log(
                    f"[media] Nem sikerült elindítani a hanglejátszást: {exc}",
                    logging.ERROR,
                )
                self._audio_sink = None
        elif self._audio_sink:
            self._audio_sink.set_volume(self._audio_volume)
        return self._audio_sink

    def set_audio_volume(self, volume: float) -> None:
        self._audio_volume = float(max(0.0, min(1.0, volume)))
        if self._audio_sink:
            self._audio_sink.set_volume(self._audio_volume)

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
            if m == "CURSOR_HIDE":
                self._on_remote_pointer(0.0, 0.0, False)
                return
            if m.startswith("CURSOR"):
                parts = m.split()
                if len(parts) >= 3:
                    try:
                        norm_x = float(parts[1])
                        norm_y = float(parts[2])
                    except ValueError:
                        self._emit_log(f"Érvénytelen kurzor üzenet: {m}", logging.DEBUG)
                        return
                    visible = True
                    if len(parts) >= 4:
                        flag = parts[3].lower()
                        visible = flag not in {"0", "false", "off"}
                    self._on_remote_pointer(norm_x, norm_y, visible)
                else:
                    self._emit_log(f"Hiányos kurzor üzenet: {m}", logging.DEBUG)
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

    async def _ws_loop(self, ws):
        reason: Optional[str] = None
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                data = json.loads(m.data)
                op = data.get("op")
                if op == "offer" and self.role == "receiver":
                    if not self.pc:
                        continue
                    sdp = data["sdp"]
                    await self.pc.setRemoteDescription(RTCSessionDescription(**sdp))
                    ans = await self.pc.createAnswer()
                    await self.pc.setLocalDescription(ans)
                    await wait_ice_complete(self.pc)
                    await ws.send_json({"op":"answer","sdp":{
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
            reason = f"WS loop ended: {e}"
            self._emit_log(reason, logging.ERROR)
        finally:
            if not self._stopping and self.ws is ws:
                msg = reason or "A jelzéscsatorna lezárult, újracsatlakozás szükséges."
                self._schedule_reconnect(msg, min_delay=3.0)

    def _cancel_scheduled_reconnect(self) -> Optional[asyncio.Task]:
        task = self._reconnect_task
        if task and not task.done():
            task.cancel()
        self._reconnect_task = None
        return task

    def _compute_reconnect_delay(self, attempt: int, min_delay: Optional[float]) -> float:
        if attempt <= 0:
            attempt = 1
        if attempt <= len(self._backoff_steps):
            delay = self._backoff_steps[attempt - 1]
        else:
            delay = self._max_backoff
        if min_delay is not None:
            delay = max(delay, float(min_delay))
        if delay < 0.0:
            delay = 0.0
        if delay > self._max_backoff and (
            min_delay is None or float(min_delay) <= self._max_backoff
        ):
            delay = self._max_backoff
        return delay

    def _schedule_reconnect(self, reason: str, *, min_delay: Optional[float] = None) -> None:
        if self._stopping:
            return
        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_attempt += 1
        delay = self._compute_reconnect_delay(self._reconnect_attempt, min_delay)
        display_delay = f"{delay:.1f}" if delay < 1.0 else f"{delay:.0f}"
        base_reason = str(reason) if reason is not None else ""
        clean_reason = " ".join(base_reason.strip().split()) if base_reason else "Kapcsolat megszakadt."
        message = f"{clean_reason} Újrapróbálkozás {display_delay} másodperc múlva..."
        self._emit_log(message, logging.WARNING)
        self.status.emit(message)
        self._cancel_scheduled_reconnect()
        loop = asyncio.get_running_loop()
        self._reconnect_task = loop.create_task(self._reconnect_after(delay))

    async def _reconnect_after(self, delay: float):
        try:
            if delay > 0:
                loop = asyncio.get_running_loop()
                end_at = loop.time() + delay
                while not self._stopping:
                    remaining = end_at - loop.time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(1.0, remaining))
                if self._stopping:
                    return
            if self._stopping:
                return
            self.status.emit("Újracsatlakozás...")
            await self._restart_connection()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._emit_log(f"Újracsatlakozás sikertelen: {e}", logging.ERROR)
            if not self._stopping:
                self._schedule_reconnect(f"Újracsatlakozás sikertelen: {e}")
        finally:
            if self._reconnect_task is asyncio.current_task():
                self._reconnect_task = None

    async def _restart_connection(self) -> None:
        async with self._restart_lock:
            if self._stopping:
                return
            self._emit_log("Újracsatlakozási kísérlet folyamatban...")
            await self._cleanup_connection(for_reconnect=True)
            if self._stopping:
                return
            self._set_state(ConnectionState.CONNECTING)
            await self._establish_connection()
            self._mark_connected()
            self._emit_log("Kapcsolat helyreállt.")
            self.status.emit("Kapcsolat helyreállt ✅")
            params = self._share_resume_params
            if params:
                self._emit_log("[media] Képernyőmegosztás automatikus újraindítása.")
                try:
                    await self.start_share(
                        params.width,
                        params.height,
                        params.fps,
                        params.bitrate_kbps,
                        params.share_audio,
                    )
                except Exception as e:
                    self._emit_log(
                        f"[media] Automatikus képernyőmegosztás indítása sikertelen: {e}",
                        logging.ERROR,
                    )

    async def _cleanup_connection(self, *, for_reconnect: bool) -> None:
        try:
            await self.stop_share(renegotiate=False, for_reconnect=for_reconnect)
        except Exception:
            pass

        ws = self.ws
        self.ws = None
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

        session = self.session
        self.session = None
        if session:
            try:
                await session.close()
            except Exception:
                pass

        pc = self.pc
        self.pc = None
        if pc:
            try:
                await pc.close()
            except Exception:
                pass

        self.channel = None
        self._clear_pending_answer(cancel=True)
        self._cancel_pointer_hide()
        self.pointer.emit(0.0, 0.0, False)
        self._cancel_ice_retry()

        ws_task = self._ws_task
        self._ws_task = None
        if ws_task:
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._emit_log(f"WS loop lezárása hibával: {e}", logging.DEBUG)

        if not for_reconnect:
            self._share_resume_params = None
            self._reconnect_attempt = 0
            self._set_state(ConnectionState.DISCONNECTED)
        else:
            if not self._stopping:
                self._set_state(ConnectionState.RECONNECTING)

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

    async def send_pointer(self, norm_x: float, norm_y: float) -> None:
        if not self.channel or self.channel.readyState != "open":
            self._emit_log("Kurzor pozíció nem küldhető: a DataChannel zárva.", logging.WARNING)
            return
        try:
            nx = float(norm_x)
            ny = float(norm_y)
        except (TypeError, ValueError):
            self._emit_log("Kurzor pozíció érvénytelen.", logging.DEBUG)
            return
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        message = f"CURSOR {nx:.4f} {ny:.4f}"
        try:
            self.channel.send(message)
        except Exception as exc:
            self._emit_log(f"Kurzor pozíció küldési hiba: {exc}", logging.WARNING)

    async def send_pointer_hide(self) -> None:
        if not self.channel or self.channel.readyState != "open":
            return
        try:
            self.channel.send("CURSOR_HIDE")
        except Exception as exc:
            self._emit_log(f"Kurzor elrejtési üzenet küldési hiba: {exc}", logging.DEBUG)

    async def start_share(
        self,
        width: int,
        height: int,
        fps: int,
        bitrate_kbps: int,
        share_audio: bool,
    ):
        if not self.pc:
            self._emit_log("Nincs aktív PeerConnection.", logging.WARNING)
            return
        if self._share_track is not None:
            self._emit_log("Már fut a megosztás.", logging.WARNING)
            return

        self._share_track = ScreenShareTrack(width=width, height=height, fps=fps)
        self._video_sender = self.pc.addTrack(self._share_track)

        audio_enabled = False
        if share_audio:
            if not audio_capture_supported():
                self._emit_log(
                    "[media] A rendszerhang megosztása nem támogatott ezen az eszközön.",
                    logging.WARNING,
                )
            else:
                try:
                    self._audio_track = SystemAudioTrack()
                    self._audio_sender = self.pc.addTrack(self._audio_track)
                    self._emit_log("[media] Hangmegosztás engedélyezve.")
                    audio_enabled = True
                except Exception as exc:
                    self._emit_log(
                        f"[media] Hangmegosztás indítása sikertelen: {exc}",
                        logging.ERROR,
                    )
                    self._audio_track = None
                    self._audio_sender = None
        else:
            await self._stop_audio_sender()

        audio_enabled = audio_enabled or bool(self._audio_track)

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
                self._emit_log(
                    "[media] Az aiortc verziód nem támogatja a setParameters-t (bitráta/fps encoder-szinten).",
                    logging.WARNING,
                )
        except Exception as e:
            self._emit_log(f"[media] setParameters hiba (indítás): {e}", logging.ERROR)

        params_snapshot = ShareParams(width, height, fps, bitrate_kbps, audio_enabled)
        await self._create_and_send_offer()
        self._share_resume_params = params_snapshot
        self._emit_log("[media] Képernyőmegosztás elindítva.")

    async def stop_share(self, *, renegotiate: bool = True, for_reconnect: bool = False):
        if self._video_sender:
            if self.pc:
                try:
                    self.pc.removeTrack(self._video_sender)
                except Exception:
                    pass
            self._video_sender = None
        if self._share_track:
            try:
                await self._share_track.stop()
            except Exception:
                pass
            self._share_track = None
        await self._stop_audio_sender()
        if renegotiate and self.pc:
            try:
                await self._create_and_send_offer()
            except Exception as e:
                self._emit_log(f"[media] renegotiation hiba (stop_share): {e}", logging.ERROR)
        if not for_reconnect:
            self._share_resume_params = None
        self._emit_log("[media] Képernyőmegosztás leállítva.")
        self._cancel_pointer_hide()

    async def _stop_audio_sender(self) -> None:
        if self._audio_sender:
            if self.pc:
                try:
                    self.pc.removeTrack(self._audio_sender)
                except Exception:
                    pass
            self._audio_sender = None
        if self._audio_track:
            try:
                await self._audio_track.stop()
            except Exception:
                pass
            self._audio_track = None

    async def set_resolution(self, width: int, height: int):
        if self._share_track:
            self._share_track.set_size(width, height)
            self._emit_log(f"[media] Új felbontás: {width}×{height}")
        if self._share_resume_params:
            params = self._share_resume_params
            self._share_resume_params = ShareParams(
                width,
                height,
                params.fps,
                params.bitrate_kbps,
                params.share_audio,
            )

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
        if self._share_resume_params:
            params = self._share_resume_params
            self._share_resume_params = ShareParams(
                params.width,
                params.height,
                fps,
                params.bitrate_kbps,
                params.share_audio,
            )

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
        if self._share_resume_params:
            params = self._share_resume_params
            self._share_resume_params = ShareParams(
                params.width,
                params.height,
                params.fps,
                kbps,
                params.share_audio,
            )

    async def stop(self):
        self._stopping = True
        task = self._cancel_scheduled_reconnect()
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._cleanup_connection(for_reconnect=False)
        self.status.emit("Leállítva")
        if self._audio_sink:
            try:
                self._audio_sink.close()
            except Exception:
                pass
            self._audio_sink = None

    def _cancel_ice_retry(self) -> None:
        task = self._ice_retry_task
        if task and not task.done():
            task.cancel()
        self._ice_retry_task = None

    def _ensure_ice_retry(self, *, initial_delay: float = 0.0, force_restart: bool = False) -> None:
        if self._stopping:
            return
        if self._ice_retry_task and not self._ice_retry_task.done():
            if force_restart:
                self._ice_retry_task.cancel()
            else:
                return
        pc = self.pc
        if pc is None:
            return
        loop = asyncio.get_running_loop()
        self._ice_retry_task = loop.create_task(
            self._ice_retry_loop(pc, initial_delay=initial_delay)
        )

    async def _ice_retry_loop(self, pc: RTCPeerConnection, *, initial_delay: float = 0.0) -> None:
        attempt = 0
        try:
            if initial_delay > 0:
                try:
                    await asyncio.sleep(initial_delay)
                except asyncio.CancelledError:
                    raise
            while not self._stopping and self.pc is pc:
                state = pc.iceConnectionState
                if state in {"connected", "completed"}:
                    break
                attempt += 1
                if attempt > self._max_ice_restart_attempts:
                    self._emit_log(
                        "ICE újrapróbálkozás: a maximális próbálkozásszám kimerült, teljes újracsatlakozás indítása.",
                        logging.WARNING,
                    )
                    self._schedule_reconnect(
                        "ICE újrapróbálkozás: a maximális próbálkozásszám kimerült.",
                        min_delay=1.0,
                    )
                    break
                if self.role != "sender":
                    self._emit_log(
                        "ICE újrapróbálkozás kihagyva – csak a küldő kezdeményezhet ICE restartot.",
                        logging.DEBUG,
                    )
                    break
                try:
                    await self._perform_ice_restart(pc, attempt)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._emit_log(
                        f"ICE újrapróbálkozás #{attempt} sikertelen: {exc}",
                        logging.WARNING,
                    )
                await asyncio.sleep(min(10.0, 2.0 * attempt))
        finally:
            if self._ice_retry_task is asyncio.current_task():
                self._ice_retry_task = None

    async def _perform_ice_restart(self, pc: RTCPeerConnection, attempt: int) -> None:
        if self.ws is None:
            raise RuntimeError("ICE újrapróbálkozás: a jelzéscsatorna nem elérhető.")
        await self._wait_for_pending_answer()
        async with self._offer_lock:
            if self.pc is not pc:
                return
            state = getattr(pc, "signalingState", None)
            if state and state != "stable":
                self._emit_log(
                    "ICE restart nem indítható – a jelzési állapot nem stabil.",
                    logging.DEBUG,
                )
                return
            self._emit_log(f"ICE újrapróbálkozás #{attempt} indítása...", logging.INFO)
            offer = await pc.createOffer(iceRestart=True)
            await pc.setLocalDescription(offer)
            await wait_ice_complete(pc)
            await self.ws.send_json(
                {
                    "op": "offer",
                    "sdp": {
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp,
                    },
                }
            )

    def _on_remote_pointer(self, norm_x: float, norm_y: float, visible: bool) -> None:
        nx = max(0.0, min(1.0, float(norm_x)))
        ny = max(0.0, min(1.0, float(norm_y)))
        if self._share_track:
            if visible:
                self._share_track.set_remote_pointer(nx, ny)
            else:
                self._share_track.clear_remote_pointer()
        self.pointer.emit(nx, ny, visible)
        if visible:
            self._schedule_pointer_hide()
        else:
            self._cancel_pointer_hide()

    def _schedule_pointer_hide(self, delay: float = 3.0) -> None:
        loop = asyncio.get_running_loop()
        task = self._pointer_hide_task
        if task and not task.done():
            task.cancel()
        self._pointer_hide_task = loop.create_task(self._hide_pointer_after(delay))

    def _cancel_pointer_hide(self) -> None:
        task = self._pointer_hide_task
        if task and not task.done():
            task.cancel()
        self._pointer_hide_task = None

    async def _hide_pointer_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._pointer_hide_task = None
        if self._share_track:
            self._share_track.clear_remote_pointer()
        self.pointer.emit(0.0, 0.0, False)

    def get_capture_geometry(self) -> Optional[Tuple[int, int, int, int]]:
        if self._share_track:
            return self._share_track.get_last_capture_bbox()
        return None

    def update_local_pointer(
        self,
        norm_x: float,
        norm_y: float,
        visible: bool,
        shape: Optional[str] = None,
        cursor_handle: Optional[int] = None,
    ) -> None:
        track = self._share_track
        if not track:
            return
        if visible:
            track.set_local_pointer(norm_x, norm_y, shape=shape, cursor_handle=cursor_handle)
        else:
            track.clear_local_pointer()
