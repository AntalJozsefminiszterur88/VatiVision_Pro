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
from functools import lru_cache
from typing import Optional

import numpy as np
import av
from aiortc import AudioStreamTrack

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - handled gracefully
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyaudio
except Exception:  # pragma: no cover - handled gracefully
    pyaudio = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
except Exception:  # pragma: no cover - handled gracefully
    AudioUtilities = None  # type: ignore
    IAudioMeterInformation = None  # type: ignore

logger = logging.getLogger(__name__)


_SAMPLE_RATE = 48_000
_CHANNELS = 1
_BLOCK_DURATION = 0.02  # 20 ms


def _downmix_to_stereo(data: np.ndarray) -> np.ndarray:
    """Ensure the provided audio data is stereo."""

    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    channels = data.shape[1] if data.ndim > 1 else 1
    if channels == 2:
        return data
    if channels <= 0:
        return np.zeros((data.shape[0], 2), dtype=data.dtype)
    if channels == 1:
        return np.repeat(data, 2, axis=1)

    mono = data.mean(axis=1, keepdims=True, dtype=data.dtype)
    return np.repeat(mono, 2, axis=1)


def _find_sounddevice_loopback_device() -> Optional[int]:  # pragma: no cover - thin wrapper
    if sd is None:
        return None

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
        if idx >= 0 and _supports_loopback(idx):
            return idx

    try:
        devices = sd.query_devices()
    except Exception:
        logger.debug("Nem sikerült lekérdezni a sounddevice eszközöket.", exc_info=True)
        return None

    for idx, _ in enumerate(devices):
        if _supports_loopback(idx):
            return idx

    return None


def _sounddevice_capture_supported() -> bool:
    if sd is None or not sys.platform.startswith("win"):
        return False
    return _find_sounddevice_loopback_device() is not None


@lru_cache(maxsize=1)
def _pyaudio_loopback_device_index() -> Optional[int]:
    """Return the PyAudio device index capable of WASAPI loopback capture."""

    if pyaudio is None or not sys.platform.startswith("win"):
        return None

    try:
        pa = pyaudio.PyAudio()
    except Exception:
        logger.debug("Nem sikerült inicializálni a PyAudio-t.", exc_info=True)
        return None

    try:
        try:
            host_api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception:
            logger.debug("A PyAudio nem támogatja a WASAPI host API-t.", exc_info=True)
            return None

        wasapi_index = host_api.get("index")
        if wasapi_index is None:
            return None

        try:
            default_loopback = pa.get_default_wasapi_loopback_device()
        except Exception:
            default_loopback = None

        if default_loopback is not None:
            device_index = default_loopback.get("index")
        else:
            device_index = None
            for idx in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(idx)
                except Exception:
                    continue
                if info.get("hostApi") != wasapi_index:
                    continue
                if info.get("isLoopbackDevice"):
                    device_index = idx
                    break

        if device_index is None:
            logger.debug("PyAudio: nem található loopback eszköz.")
            return None

        try:
            pa.is_format_supported(
                rate=_SAMPLE_RATE,
                input_device=device_index,
                input_channels=_CHANNELS,
                input_format=pyaudio.paFloat32,
            )
        except Exception:
            logger.debug(
                "PyAudio: a kiválasztott eszköz nem támogatja a kívánt formátumot.",
                exc_info=True,
            )
            return None

        return int(device_index)
    finally:
        try:
            pa.terminate()
        except Exception:
            pass


def _pyaudio_capture_supported() -> bool:
    return _pyaudio_loopback_device_index() is not None


def audio_capture_supported() -> bool:
    """Return True if loopback capture is supported on this platform."""

    return _sounddevice_capture_supported() or _pyaudio_capture_supported()


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
        self._stream_lock = threading.Lock()
        self._sd_stream: Optional[sd.InputStream] = None
        self._pa: Optional[pyaudio.PyAudio] = None  # type: ignore[assignment]
        self._pa_stream: Optional[object] = None
        self._backend: Optional[str] = None

        errors: list[str] = []
        if _sounddevice_capture_supported():
            try:
                self._start_sounddevice_stream()
                self._backend = "sounddevice"
            except Exception as exc:
                errors.append(f"sounddevice: {exc}")
        if self._backend is None and _pyaudio_capture_supported():
            try:
                self._start_pyaudio_stream()
                self._backend = "pyaudio"
            except Exception as exc:
                errors.append(f"pyaudio: {exc}")

        if self._backend is None:
            detail = f" Részletek: {'; '.join(errors)}" if errors else ""
            raise RuntimeError("Nem sikerült elindítani a rendszerhang megosztását." + detail)

    # ------------------------------------------------------------------ helpers
    def _start_sounddevice_stream(self) -> None:
        if sd is None:
            raise RuntimeError("A sounddevice modul nem érhető el.")
        device = _find_sounddevice_loopback_device()
        if device is None:
            raise RuntimeError(
                "Nem található WASAPI kompatibilis hangeszköz a rendszerhang megosztásához."
            )
        try:
            wasapi_settings = sd.WasapiSettings(loopback=True)
        except Exception:
            wasapi_settings = None
        def _query_device(kind: Optional[str] = None) -> dict[str, float | int]:
            try:
                info = sd.query_devices(device, kind) if kind else sd.query_devices(device)
            except Exception:
                return {}
            if isinstance(info, dict):
                return info
            # sounddevice can also return an object with attributes
            extracted: dict[str, float | int] = {}
            for key in ("max_input_channels", "max_output_channels", "default_samplerate"):
                value = getattr(info, key, None)
                if value is not None:
                    extracted[key] = value
            return extracted

        info_default = _query_device()
        info_input = _query_device("input")
        info_output = _query_device("output")

        logger.debug(
            "SoundDevice loopback eszközinformáció: default=%s input=%s output=%s",
            info_default,
            info_input,
            info_output,
        )

        possible_channels: list[int] = []
        for info in (info_default, info_input, info_output):
            for key in ("max_input_channels", "max_output_channels"):
                value = info.get(key)
                if value:
                    possible_channels.append(int(value))
        possible_channels.extend([_CHANNELS, 2, 1])

        possible_samplerates: list[int] = []
        for info in (info_default, info_input, info_output):
            samplerate = info.get("default_samplerate")
            if samplerate:
                try:
                    possible_samplerates.append(int(round(float(samplerate))))
                except Exception:
                    continue
        possible_samplerates.extend([_SAMPLE_RATE, 48_000, 44_100])

        attempt_errors: list[str] = []
        last_error: Optional[Exception] = None

        def _unique(values: list[int]) -> list[int]:
            seen: set[int] = set()
            ordered: list[int] = []
            for value in values:
                if not value:
                    continue
                if value in seen:
                    continue
                seen.add(value)
                ordered.append(value)
            return ordered

        channels_to_try = _unique(possible_channels)
        samplerates_to_try = _unique(possible_samplerates)

        for channels in channels_to_try:
            for samplerate in samplerates_to_try:
                try:
                    logger.debug(
                        "SoundDevice loopback próbálkozás: device=%s channels=%s samplerate=%s",
                        device,
                        channels,
                        samplerate,
                    )
                    stream = sd.InputStream(
                        samplerate=samplerate,
                        channels=channels,
                        blocksize=self._block_frames,
                        dtype="float32",
                        device=device,
                        callback=self._on_sounddevice_audio,
                        extra_settings=wasapi_settings,
                    )
                    stream.start()
                except Exception as exc:  # pragma: no cover - runtime guard
                    last_error = exc
                    attempt_errors.append(
                        f"channels={channels}, samplerate={samplerate}: {exc}"
                    )
                    continue
                self._sd_stream = stream
                logger.debug(
                    "SoundDevice loopback sikeresen elindítva: device=%s channels=%s samplerate=%s",
                    device,
                    channels,
                    samplerate,
                )
                return

        if last_error is not None:
            logger.error(
                "Nem sikerült elindítani a rendszerhang rögzítését. Próbálkozások: %s",
                "; ".join(attempt_errors),
            )
            raise last_error

        raise RuntimeError("Nem sikerült elindítani a rendszerhang rögzítését.")

    def _start_pyaudio_stream(self) -> None:
        if pyaudio is None:
            raise RuntimeError("A PyAudio modul nem érhető el.")
        device_index = _pyaudio_loopback_device_index()
        if device_index is None:
            raise RuntimeError(
                "Nem található WASAPI loopback eszköz a PyAudio számára."
            )
        try:
            stream_info = pyaudio.PaWasapiStreamInfo(flags=pyaudio.paWinWasapiLoopback)
        except Exception as exc:
            raise RuntimeError("A PyAudio nem támogatja a WASAPI loopback módot.") from exc

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=_CHANNELS,
                rate=_SAMPLE_RATE,
                frames_per_buffer=self._block_frames,
                input=True,
                input_device_index=device_index,
                stream_callback=self._on_pyaudio_audio,
                start=False,
                stream_info=stream_info,
            )
            stream.start_stream()
        except Exception:
            try:
                pa.terminate()
            except Exception:
                pass
            raise
        self._pa = pa
        self._pa_stream = stream

    def _submit_audio_block(self, data: np.ndarray) -> None:
        def _put() -> None:
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

    # ------------------------------------------------------------------ callbacks
    def _on_sounddevice_audio(self, indata, frames, time_info, status):  # pragma: no cover - realtime callback
        if status:
            logger.debug("Rendszerhang stream státusz (sounddevice): %s", status)
        if self._closing:
            return
        data = np.array(indata, dtype=np.float32)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        if data.shape[1] != _CHANNELS:
            data = _downmix_to_stereo(data)
        if self._discord_meter.is_active():
            data = np.zeros_like(data)
        self._submit_audio_block(data.T)

    def _on_pyaudio_audio(self, in_data, frame_count, time_info, status_flags):  # pragma: no cover - realtime callback
        status_value = pyaudio.paContinue if pyaudio is not None else 0
        if status_flags:
            logger.debug("Rendszerhang stream státusz (PyAudio): %s", status_flags)
        if self._closing or pyaudio is None:
            return (None, status_value)
        data = np.frombuffer(in_data, dtype=np.float32)
        if not data.size:
            return (None, status_value)
        try:
            data = data.reshape(-1, _CHANNELS)
        except ValueError:
            data = data[: frame_count * _CHANNELS].reshape(-1, _CHANNELS)
        if data.shape[1] != _CHANNELS:
            data = _downmix_to_stereo(data)
        if self._discord_meter.is_active():
            data = np.zeros_like(data)
        self._submit_audio_block(data.T)
        return (None, status_value)

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
            sd_stream = self._sd_stream
            self._sd_stream = None
            pa_stream = self._pa_stream
            self._pa_stream = None
            pa_instance = self._pa
            self._pa = None
        if sd_stream is not None:
            try:
                sd_stream.stop()
            except Exception:
                pass
            try:
                sd_stream.close()
            except Exception:
                pass
        if pa_stream is not None and pyaudio is not None:
            try:
                pa_stream.stop_stream()
            except Exception:
                pass
            try:
                pa_stream.close()
            except Exception:
                pass
        if pa_instance is not None:
            try:
                pa_instance.terminate()
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
