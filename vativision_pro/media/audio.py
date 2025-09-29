"""Audio utilities for capturing and playing system audio."""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np
import av
from aiortc import AudioStreamTrack

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - handled gracefully
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
except Exception:  # pragma: no cover - handled gracefully
    AudioUtilities = None  # type: ignore
    IAudioMeterInformation = None  # type: ignore

logger = logging.getLogger(__name__)


_SAMPLE_RATE = 48_000
_CHANNELS = 2
_BLOCK_DURATION = 0.02  # 20 ms


def audio_capture_supported() -> bool:
    """Return True if loopback capture is supported on this platform."""

    return bool(sd) and sys.platform.startswith("win")


def audio_playback_supported() -> bool:
    """Return True if audio playback is supported on this platform."""

    return bool(sd)


@dataclass
class _DiscordMeter:
    """Helper that checks whether Discord currently plays audio."""

    threshold: float = 1e-3
    refresh_interval: float = 0.25
    _last_poll: float = 0.0
    _active: bool = False

    def is_active(self) -> bool:
        if AudioUtilities is None or IAudioMeterInformation is None:
            return False
        now = time.monotonic()
        if now - self._last_poll < self.refresh_interval:
            return self._active
        self._last_poll = now
        try:  # pragma: no cover - system integration
            sessions = AudioUtilities.GetAllSessions()
        except Exception:
            logger.debug("Nem sikerült lekérdezni az audio session-öket.", exc_info=True)
            return False
        active = False
        for session in sessions:
            process = getattr(session, "Process", None)
            if process is None:
                continue
            try:
                name = process.name()
            except Exception:  # pragma: no cover - defensive
                continue
            if not name:
                continue
            if name.lower() != "discord.exe":
                continue
            try:
                meter = session._ctl.QueryInterface(IAudioMeterInformation)
                level = meter.GetPeakValue()
            except Exception:
                logger.debug(
                    "Nem sikerült lekérdezni a Discord hangerő mérőjét.",
                    exc_info=True,
                )
                continue
            if level >= self.threshold:
                active = True
                break
        self._active = active
        return active


class SystemAudioTrack(AudioStreamTrack):
    """AudioStreamTrack that captures system audio via WASAPI loopback."""

    def __init__(self, *, block_duration: float = _BLOCK_DURATION):
        if not audio_capture_supported():  # pragma: no cover - platform guard
            raise RuntimeError("A rendszerhang megosztás nem támogatott ezen a platformon.")
        super().__init__()
        self._loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=8)
        self._pts = 0
        self._block_frames = int(_SAMPLE_RATE * block_duration)
        self._discord_meter = _DiscordMeter()
        self._closing = False
        self._stream: Optional[sd.InputStream] = None
        self._stream_lock = threading.Lock()
        self._start_stream()

    # ------------------------------------------------------------------ utils
    def _find_loopback_device(self) -> Optional[int]:  # pragma: no cover - thin wrapper
        assert sd is not None

        def _supports_loopback(idx: int) -> bool:
            try:
                info = sd.query_devices(idx)
            except Exception:
                return False
            if info.get("max_output_channels", 0) < 1:
                return False
            hostapi_index = info.get("hostapi")
            hostapi_name = ""
            if hostapi_index is not None:
                try:
                    hostapi = sd.query_hostapis(hostapi_index)
                    hostapi_name = str(hostapi.get("name", ""))
                except Exception:
                    hostapi_name = ""
            return "wasapi" in hostapi_name.lower()

        try:
            default_output = sd.default.device[1]
        except Exception:
            default_output = None

        if isinstance(default_output, (list, tuple)):
            default_output = default_output[0] if default_output else None

        if isinstance(default_output, (int, float)):
            idx = int(default_output)
            if idx >= 0:
                if _supports_loopback(idx):
                    return idx

        devices = sd.query_devices()
        for idx, _ in enumerate(devices):
            if _supports_loopback(idx):
                return idx

        return None

    def _start_stream(self) -> None:
        assert sd is not None
        device = self._find_loopback_device()
        if device is None:
            raise RuntimeError(
                "Nem található WASAPI kompatibilis hangeszköz a rendszerhang megosztásához."
            )
        wasapi_settings = None
        try:
            wasapi_settings = sd.WasapiSettings(loopback=True)
        except Exception:
            wasapi_settings = None
        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                blocksize=self._block_frames,
                dtype="float32",
                device=device,
                callback=self._on_audio,
                extra_settings=wasapi_settings,
            )
            stream.start()
            self._stream = stream
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("Nem sikerült elindítani a rendszerhang rögzítését: %s", exc)
            raise

    def _on_audio(self, indata, frames, time_info, status):  # pragma: no cover - realtime callback
        if status:
            logger.debug("Rendszerhang stream státusz: %s", status)
        if self._closing:
            return
        data = np.array(indata, dtype=np.float32)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        if self._discord_meter.is_active():
            data = np.zeros_like(data)
        data = data.T  # convert to (channels, samples)

        def _put():
            if self._closing:
                return
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(data)

        try:
            self._loop.call_soon_threadsafe(_put)
        except RuntimeError:
            # event loop already closed
            pass

    async def recv(self) -> av.AudioFrame:
        data = await self._queue.get()
        frame = av.AudioFrame.from_ndarray(data, format="flt", layout="stereo")
        frame.sample_rate = _SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, _SAMPLE_RATE)
        self._pts += data.shape[1]
        return frame

    async def stop(self) -> None:
        self._closing = True
        with self._stream_lock:
            stream = self._stream
            self._stream = None
        if stream:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        await super().stop()


class AudioPlayback:
    """Simple playback helper used by the receiver side."""

    def __init__(self, volume: float = 1.0):
        if not audio_playback_supported():  # pragma: no cover - platform guard
            raise RuntimeError("Hanglejátszás nem támogatott ezen a platformon.")
        assert sd is not None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._current: Optional[np.ndarray] = None
        self._offset = 0
        self._volume = float(max(0.0, min(1.0, volume)))
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            blocksize=int(_SAMPLE_RATE * _BLOCK_DURATION),
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    # ------------------------------------------------------------------ helpers
    def submit(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32, copy=False)
        with self._lock:
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(data)

    def _pull_chunk(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._current is None or self._offset >= self._current.shape[1]:
                try:
                    self._current = self._queue.get_nowait()
                except queue.Empty:
                    self._current = None
                    self._offset = 0
                    return None
                self._offset = 0
            chunk = self._current[:, self._offset :]
            self._offset = self._current.shape[1]
            return chunk

    def _on_audio(self, outdata, frames, time_info, status):  # pragma: no cover - realtime callback
        if status:
            logger.debug("Lejátszási stream státusz: %s", status)
        needed = frames
        chunks = []
        while needed > 0:
            chunk = self._pull_chunk()
            if chunk is None:
                break
            take = min(needed, chunk.shape[1])
            chunks.append(chunk[:, :take])
            remaining = chunk[:, take:]
            if remaining.size:
                with self._lock:
                    self._current = remaining
                    self._offset = 0
            else:
                with self._lock:
                    self._current = None
                    self._offset = 0
            needed -= take
        if chunks:
            merged = np.concatenate(chunks, axis=1)
            if merged.shape[1] < frames:
                pad = np.zeros((merged.shape[0], frames - merged.shape[1]), dtype=np.float32)
                merged = np.concatenate([merged, pad], axis=1)
            out = (merged.T) * self._volume
        else:
            out = np.zeros((frames, _CHANNELS), dtype=np.float32)
        outdata[:] = out

    def set_volume(self, volume: float) -> None:
        self._volume = float(max(0.0, min(1.0, volume)))

    def close(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        self._current = None
        self._offset = 0


__all__ = [
    "SystemAudioTrack",
    "AudioPlayback",
    "audio_capture_supported",
    "audio_playback_supported",
]
