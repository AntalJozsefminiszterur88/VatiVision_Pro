import asyncio
import logging
import queue
import sys
import threading
from fractions import Fraction
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from av.audio.frame import AudioFrame
from aiortc.mediastreams import AudioStreamTrack

try:  # pragma: no cover - fallback if aiortc changes exports
    from aiortc import MediaStreamError
except Exception:  # pragma: no cover
    class MediaStreamError(RuntimeError):
        pass


logger = logging.getLogger(__name__)


AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_BLOCK_SIZE = 1024
DISCORD_PROCESSES = {"discord.exe", "discord"}


class SystemAudioTrack(AudioStreamTrack):
    """Audio track that captures system output audio."""

    kind = "audio"

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        blocksize: int = AUDIO_BLOCK_SIZE,
    ) -> None:
        if sd is None:
            raise RuntimeError(
                "A 'sounddevice' csomag nem érhető el, a hang megosztás nem indítható."
            )
        super().__init__()
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._blocksize = int(blocksize)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - fallback for non-async contexts
            self._loop = asyncio.get_event_loop()
        self._queue: "asyncio.Queue[Optional[np.ndarray]]" = asyncio.Queue(maxsize=8)
        self._stream: Optional[sd.InputStream] = None
        self._timestamp = 0
        self._closed = asyncio.Event()
        self._start_stream()

    def _start_stream(self) -> None:
        assert sd is not None
        device = self._pick_loopback_device()

        def _callback(indata, frames, _time, status) -> None:
            if status:
                logger.warning("Loopback rögzítési státusz: %s", status)
            data = np.array(indata, dtype=np.float32, copy=True)
            try:
                self._loop.call_soon_threadsafe(self._handle_chunk, data)
            except RuntimeError:
                pass

        logger.info(
            "Hang loopback stream indítása (eszköz=%s, sr=%s, ch=%s)",
            device,
            self._sample_rate,
            self._channels,
        )
        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=_callback,
            device=device,
            latency="low",
        )
        stream.start()
        self._stream = stream

    def _handle_chunk(self, data: np.ndarray) -> None:
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.debug("Hangpuffer telített, eldobás történt.")

    def _pick_loopback_device(self) -> Optional[int]:
        assert sd is not None
        try:
            devices = sd.query_devices()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Nem sikerült lekérdezni a hang eszközöket: %s", exc)
            return None
        loopbacks: Sequence[Tuple[int, dict]] = []
        for idx, info in enumerate(devices):
            name = str(info.get("name", ""))
            host = str(info.get("hostapi", ""))
            if "loopback" in name.lower() or "loopback" in host.lower():
                if int(info.get("max_input_channels", 0)) >= self._channels:
                    loopbacks.append((idx, info))
        if loopbacks:
            idx, info = loopbacks[0]
            logger.info("Loopback eszköz kiválasztva: %s (index=%s)", info.get("name"), idx)
            return idx
        logger.info("Nem találtunk dedikált loopback eszközt, az alapértelmezett lesz használva.")
        return None

    async def recv(self) -> AudioFrame:
        if self.readyState == "ended":
            raise MediaStreamError("A hang track leállt")
        chunk = await self._queue.get()
        if chunk is None:
            raise MediaStreamError("A hang track leállt")
        samples = np.clip(chunk, -1.0, 1.0)
        frame = AudioFrame.from_ndarray(
            (samples * 32767.0).astype(np.int16),
            format="s16",
            layout="stereo" if self._channels == 2 else "mono",
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self._sample_rate)
        self._timestamp += samples.shape[0]
        return frame

    async def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        except RuntimeError:
            pass
        self._closed.set()
        stop = getattr(super(), "stop", None)
        if callable(stop):
            result = stop()
            if asyncio.iscoroutine(result):
                await result


class AudioPlaybackEngine:
    """Simple playback helper driven by a queue feeding sounddevice."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=10)
        self._volume = 0.8
        if sd is None:
            logger.warning("A hang lejátszáshoz a 'sounddevice' modul szükséges.")

    def _ensure_stream(self, sample_rate: int, channels: int) -> None:
        if sd is None:
            return
        with self._lock:
            if (
                self._stream is not None
                and self._sample_rate == sample_rate
                and self._channels == channels
            ):
                return
            if self._stream is not None:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close()
            self._sample_rate = sample_rate
            self._channels = channels
            self._queue = queue.Queue(maxsize=10)

            def _callback(outdata, frames, _time, status) -> None:
                if status:
                    logger.warning("Hang lejátszási státusz: %s", status)
                try:
                    chunk = self._queue.get_nowait()
                except queue.Empty:
                    outdata.fill(0)
                    return
                if len(chunk) < frames:
                    pad = np.zeros((frames - len(chunk), channels), dtype=np.float32)
                    chunk = np.vstack((chunk, pad))
                elif len(chunk) > frames:
                    rest = chunk[frames:]
                    chunk = chunk[:frames]
                    try:
                        self._queue.put_nowait(rest)
                    except queue.Full:
                        pass
                outdata[:] = chunk * self._volume

            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                callback=_callback,
                latency="low",
            )
            stream.start()
            self._stream = stream

    def enqueue(self, payload: bytes, sample_rate: int, channels: int) -> None:
        if sd is None:
            return
        if sample_rate <= 0 or channels <= 0:
            return
        self._ensure_stream(sample_rate, channels)
        if self._stream is None:
            return
        data = np.frombuffer(payload, dtype=np.float32)
        if channels > 1:
            data = data.reshape((-1, channels))
        else:
            data = data.reshape((-1, 1))
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            logger.debug("Hang kimeneti puffer tele, csomag eldobva.")

    def set_volume(self, volume: float) -> None:
        volume = max(0.0, min(1.0, float(volume)))
        self._volume = volume

    def reset(self) -> None:
        if sd is None:
            return
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close()
                self._stream = None
            self._sample_rate = None
            self._channels = None
            self._queue = queue.Queue(maxsize=10)


def exclude_discord_from_loopback() -> bool:
    """Attempt to mark Discord processes as excluded from loopback capture."""

    if not sys.platform.startswith("win"):
        return False
    if psutil is None:
        logger.warning("A Discord kizárásához a 'psutil' modul szükséges.")
        return False
    excluded_any = False
    for proc in psutil.process_iter(["name", "pid"]):
        name = (proc.info.get("name") or "").lower()
        if name not in DISCORD_PROCESSES:
            continue
        pid = int(proc.info.get("pid") or 0)
        if pid <= 0:
            continue
        try:
            _set_process_loopback_mode(pid, 1)
            excluded_any = True
        except Exception as exc:  # pragma: no cover - platform specific
            logger.warning("Nem sikerült kizárni a Discord hangját (pid=%s): %s", pid, exc)
    return excluded_any


def _set_process_loopback_mode(process_id: int, mode: int) -> None:
    """Call into the Windows audio policy API to set loopback mode."""

    if not sys.platform.startswith("win"):
        raise RuntimeError("Loopback mód csak Windows rendszeren érhető el.")

    # Late imports to avoid Windows-only dependencies on other platforms.
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

        def __init__(self, guid_str: str):  # type: ignore[override]
            super().__init__()
            parts = guid_str.strip("{}" ).split("-")
            self.Data1 = int(parts[0], 16)
            self.Data2 = int(parts[1], 16)
            self.Data3 = int(parts[2], 16)
            data4 = bytes.fromhex(parts[3] + parts[4])
            for i in range(8):
                self.Data4[i] = data4[i]

    CLSCTX_ALL = 23
    CLSID_AudioPolicyConfigFactory = GUID("{294935CE-F637-4E7C-A41B-AB255460B862}")
    IID_IAudioPolicyConfigFactory = GUID("{0AFC5D8D-31B0-4B12-A015-9E50CC2F5595}")

    ole32 = ctypes.OleDLL("ole32.dll")
    hr = ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    if hr not in (0, 1):  # S_OK or S_FALSE
        raise ctypes.WinError(hr)

    factory_ptr = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        ctypes.byref(CLSID_AudioPolicyConfigFactory),
        None,
        CLSCTX_ALL,
        ctypes.byref(IID_IAudioPolicyConfigFactory),
        ctypes.byref(factory_ptr),
    )
    if hr != 0:
        raise ctypes.WinError(hr)

    vtable = ctypes.cast(factory_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents

    # The SetProcessLoopbackMode method is located at index 13 of the vtable
    # (after the standard IUnknown methods and existing policy calls).
    SetProcessLoopbackMode = ctypes.WINFUNCTYPE(
        wintypes.HRESULT, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD
    )(vtable[13])

    hr = SetProcessLoopbackMode(factory_ptr, wintypes.DWORD(process_id), wintypes.DWORD(mode))
    if hr != 0:
        raise ctypes.WinError(hr)

    Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    Release(factory_ptr)
    ole32.CoUninitialize()
