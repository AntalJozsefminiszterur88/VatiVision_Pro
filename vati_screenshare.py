import asyncio
import concurrent.futures
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
        self._monitors_cache = []
        self._capture_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vati-screenshare"
        )
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
        self._monitors_cache = []
        try:
            with mss() as temp_sct:
                self._monitors_cache = list(temp_sct.monitors)
        except Exception as exc:
            logger.exception("Nem sikerült inicializálni az MSS rögzítőt: %s", exc)
            self._monitors_cache = []
        self._monitor_index = self._sanitize_monitor_index(self._requested_monitor)

    def _sanitize_monitor_index(self, monitor_index: int, monitors=None) -> int:
        if monitors is None:
            monitors = self._monitors_cache
        monitors = monitors or []
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
        if self._sct is None:
            self._sct = mss()

        monitors = getattr(self._sct, "monitors", []) or []
        if not monitors:
            raise RuntimeError("Nem található monitor a képernyőmegosztáshoz.")

        if monitors != self._monitors_cache:
            self._monitors_cache = list(monitors)

        if self._monitor_index >= len(monitors):
            self._monitor_index = self._sanitize_monitor_index(self._monitor_index, monitors)

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

        try:
            raw = self._sct.grab(bbox, include_cursor=True)
        except TypeError:
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
            loop = asyncio.get_running_loop()
            if not self._capture_executor:
                raise RuntimeError("A képernyő rögzítő végrehajtó le lett állítva.")
            frame = await loop.run_in_executor(self._capture_executor, self._grab_frame)
        except Exception as exc:
            logger.exception("Képernyőkép készítése sikertelen: %s", exc)
            w, h = self._size or (640, 360)
            black = np.zeros((h, w, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(black, format="rgb24")

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def stop(self) -> None:
        await super().stop()
        loop = asyncio.get_running_loop()
        executor = self._capture_executor
        if executor:
            try:
                await loop.run_in_executor(executor, self._close_capture_resources)
            finally:
                self._capture_executor = None
                await loop.run_in_executor(None, executor.shutdown, True)

    def _close_capture_resources(self) -> None:
        if self._sct:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
