
import asyncio
import time
from dataclasses import dataclass
from typing import Tuple, Optional

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

@dataclass
class ShareConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30

class ScreenShareTrack(VideoStreamTrack):
    def __init__(self, width: int, height: int, fps: int, monitor_index: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        super().__init__()
        self._size = (int(width), int(height))
        self._fps = int(fps)
        self._monitor_index = int(monitor_index)
        self._region = region
        self._sct = mss()
        self._last_ts = 0.0

    def set_size(self, width: int, height: int):
        self._size = (int(width), int(height))

    def set_fps(self, fps: int):
        self._fps = int(fps)

    async def recv(self) -> VideoFrame:
        fps = max(1, int(self._fps))
        period = 1.0 / float(fps)
        now = time.perf_counter()
        wait = self._last_ts + period - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_ts = time.perf_counter()

        mon = self._sct.monitors[self._monitor_index]
        if self._region is not None:
            left, top, width, height = self._region
            bbox = {"left": mon["left"] + left, "top": mon["top"] + top, "width": width, "height": height}
        else:
            bbox = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}

        try:
            raw = self._sct.grab(bbox)        # BGRA memory; .rgb provides RGB bytes
            img = Image.frombytes("RGB", raw.size, raw.rgb)
            if self._size:
                img = img.resize(self._size, Image.BILINEAR)
            frame = VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
        except Exception:
            w, h = self._size or (640, 360)
            black = np.zeros((h, w, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(black, format="rgb24")

        frame.pts, frame.time_base = self.next_timestamp()
        return frame
