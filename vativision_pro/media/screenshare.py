import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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

# A kurzor legyen ~15%-kal kisebb, ezért csökkentjük a skálázási arányt.
CURSOR_SCALE_RATIO = 0.02125
CURSOR_MIN_SIZE = 12
CURSOR_OFFSET_X = -15
CURSOR_OFFSET_Y = -10

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
        self._cursor_image = None
        self._cursor_cache = None
        self._cursor_cache_size: Optional[Tuple[int, int]] = None
        self._pointer_visible = False
        self._pointer_norm: Tuple[float, float] = (0.0, 0.0)
        self._local_pointer_visible = False
        self._local_pointer_norm: Tuple[float, float] = (0.0, 0.0)
        self._pointer_lock = threading.Lock()
        self._last_capture_bbox: Optional[Tuple[int, int, int, int]] = None

        self._load_cursor()
        self._init_capture()

    def _load_cursor(self) -> None:
        cursor_path = Path(__file__).resolve().parent / "cursor.png"
        if not cursor_path.exists():
            logger.warning("A kurzor ikon (%s) nem található.", cursor_path)
            return
        try:
            loaded = Image.open(cursor_path).convert("RGBA")
            self._cursor_image = self._sanitize_cursor_transparency(loaded)
        except Exception as exc:
            logger.exception("Nem sikerült betölteni a kurzor ikont: %s", exc)
            self._cursor_image = None
            return
        self._cursor_cache = None
        self._cursor_cache_size = None

    @staticmethod
    def _sanitize_cursor_transparency(img: Image.Image) -> Image.Image:
        """Távolítsa el a kurzor háttér- vagy árnyék sziluettjét."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] != 4:
            return img

        rgb = arr[:, :, :3].astype(np.int16)
        alpha = arr[:, :, 3]
        background = rgb[0, 0]

        diff = np.abs(rgb - background)
        mask_background = np.all(diff <= 12, axis=-1)

        brightness = rgb.max(axis=-1)
        mask_shadow = (brightness < 40) & (alpha < 220)

        mask = mask_background | mask_shadow
        if not np.any(mask):
            return img

        arr[mask, :3] = 0
        arr[mask, 3] = 0
        return Image.fromarray(arr, mode="RGBA")

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

        self._last_capture_bbox = (
            int(bbox["left"]),
            int(bbox["top"]),
            int(bbox["width"]),
            int(bbox["height"]),
        )

        try:
            raw = self._sct.grab(bbox, include_cursor=True)
        except TypeError:
            raw = self._sct.grab(bbox)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        if self._size and raw.size != self._size:
            img = img.resize(self._size, Image.LANCZOS)
        img = self._apply_pointer_overlay(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return VideoFrame.from_image(img)

    def _get_cursor_for_size(self, frame_size: Tuple[int, int]):
        if not self._cursor_image:
            return None
        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        base = self._cursor_image
        target_width = min(base.width, max(CURSOR_MIN_SIZE, int(width * CURSOR_SCALE_RATIO)))
        aspect = base.width / base.height if base.height else 1.0
        target_height = min(base.height, max(CURSOR_MIN_SIZE, int(target_width / aspect)))
        size = (target_width, target_height)
        if self._cursor_cache is None or self._cursor_cache_size != size:
            self._cursor_cache = base.resize(size, Image.LANCZOS)
            self._cursor_cache_size = size
        return self._cursor_cache

    def _apply_pointer_overlay(self, img: Image.Image) -> Image.Image:
        cursor = self._get_cursor_for_size(img.size)
        if cursor is None:
            return img
        with self._pointer_lock:
            pointer_states = []
            if self._local_pointer_visible:
                pointer_states.append(self._local_pointer_norm)
            if self._pointer_visible:
                pointer_states.append(self._pointer_norm)
        if not pointer_states:
            return img
        width, height = img.size
        cw, ch = cursor.size
        base = img.convert("RGBA")
        for norm_x, norm_y in pointer_states:
            x = int(round(norm_x * width + CURSOR_OFFSET_X))
            y = int(round(norm_y * height + CURSOR_OFFSET_Y))
            x = max(0, min(x, max(0, width - cw)))
            y = max(0, min(y, max(0, height - ch)))
            base.paste(cursor, (x, y), cursor)
        return base

    def set_remote_pointer(self, norm_x: float, norm_y: float) -> None:
        with self._pointer_lock:
            self._pointer_visible = True
            self._pointer_norm = (float(max(0.0, min(1.0, norm_x))), float(max(0.0, min(1.0, norm_y))))

    def clear_remote_pointer(self) -> None:
        with self._pointer_lock:
            self._pointer_visible = False

    def set_local_pointer(self, norm_x: float, norm_y: float) -> None:
        with self._pointer_lock:
            self._local_pointer_visible = True
            self._local_pointer_norm = (
                float(max(0.0, min(1.0, norm_x))),
                float(max(0.0, min(1.0, norm_y))),
            )

    def clear_local_pointer(self) -> None:
        with self._pointer_lock:
            self._local_pointer_visible = False

    def get_last_capture_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_capture_bbox

    async def recv(self) -> VideoFrame:
        fps = max(1, int(self._fps))
        period = 1.0 / float(fps)
        now = time.perf_counter()
        if self._last_ts <= 0:
            self._last_ts = now
        target = self._last_ts + period
        wait = target - now
        if wait > 0:
            await asyncio.sleep(wait)
            self._last_ts = target
        else:
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
        self.clear_remote_pointer()
        self.clear_local_pointer()

    def _close_capture_resources(self) -> None:
        if self._sct:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
