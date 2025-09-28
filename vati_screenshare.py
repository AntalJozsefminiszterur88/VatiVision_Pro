import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from mss import mss
from PIL import Image
from av import VideoFrame
from aiortc import VideoStreamTrack

RESOLUTIONS = {
    "144p (256×144)":   (256, 144),
    "240p (426×240)":   (426, 240),
    "360p (640×360)":   (640, 360),
    "720p (1280×720)":  (1280, 720),
    "1080p (1920×1080)":(1920, 1080),
}

logger = logging.getLogger(__name__)


@dataclass
class ShareConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    monitor_index: int = 0
    region: Optional[Tuple[int, int, int, int]] = None


class ScreenShareTrack(VideoStreamTrack):
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        monitor_index: int = 0,
        region: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        self._size = (int(width), int(height))
        self._fps = int(fps)
        self._requested_monitor = int(monitor_index)
        self._monitor_index = 0
        self._region = region
        self._sct = None
        self._last_ts = 0.0
        self._init_capture()

    def set_size(self, width: int, height: int):
        self._size = (int(width), int(height))

    def set_fps(self, fps: int):
        self._fps = int(fps)

    def set_monitor(self, monitor_index: int):
        self._requested_monitor = int(monitor_index)
        self._monitor_index = self._sanitize_monitor_index(self._requested_monitor)

    def set_region(self, region: Optional[Tuple[int, int, int, int]]):
        self._region = region

    def _init_capture(self):
        try:
            self._sct = mss()
        except Exception as exc:
            logger.exception("Nem sikerült inicializálni az MSS rögzítőt: %s", exc)
            self._sct = None
        self._monitor_index = self._sanitize_monitor_index(self._requested_monitor)

    def _sanitize_monitor_index(self, monitor_index: int) -> int:
        if not self._sct:
            return 0
        monitors = getattr(self._sct, "monitors", []) or []
        if not monitors:
            logger.warning("Nem található egyetlen monitor sem a képernyőmegosztáshoz.")
            return 0

        idx = int(monitor_index)
        max_index = len(monitors) - 1
        if idx < 0 or idx > max_index:
            fallback = 1 if max_index >= 1 else 0
            logger.warning(
                "Érvénytelen monitor index (%s). Elérhető tartomány: 0..%s – visszaesés %s értékre.",
                idx,
                max_index,
                fallback,
            )
            idx = fallback
        return idx

    def _grab_frame(self) -> VideoFrame:
        if not self._sct:
            raise RuntimeError("A képernyő rögzítő nincs inicializálva.")

        monitors = self._sct.monitors
        if not monitors:
            raise RuntimeError("Nem található monitor a képernyőmegosztáshoz.")

        if self._monitor_index >= len(monitors):
            self._monitor_index = self._sanitize_monitor_index(self._monitor_index)

        mon = monitors[self._monitor_index]

        if self._region is not None:
            left, top, width, height = self._region
            bbox = {
                "left": int(mon["left"] + left),
                "top": int(mon["top"] + top),
                "width": int(width),
                "height": int(height),
            }
        else:
            bbox = {
                "left": int(mon["left"]),
                "top": int(mon["top"]),
                "width": int(mon["width"]),
                "height": int(mon["height"]),
            }

        raw = self._sct.grab(bbox)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        if self._size and raw.size != self._size:
            img = img.resize(self._size, Image.LANCZOS)
        return VideoFrame.from_image(img)

    async def recv(self) -> VideoFrame:
        fps = max(1, int(self._fps))
        period = 1.0 / float(fps)
        now = time.perf_counter()
        wait = self._last_ts + period - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_ts = time.perf_counter()

        try:
            frame = await asyncio.get_running_loop().run_in_executor(None, self._grab_frame)
        except Exception as exc:
            logger.exception("Képernyőkép készítése sikertelen: %s", exc)
            w, h = self._size or (640, 360)
            black = np.zeros((h, w, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(black, format="rgb24")

        frame.pts, frame.time_base = self.next_timestamp()
        return frame

    async def stop(self) -> None:
        await super().stop()
        if self._sct:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
