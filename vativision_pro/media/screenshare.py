import asyncio
import concurrent.futures
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from mss import mss
from PIL import Image
from PySide6 import QtGui
from av import VideoFrame
from aiortc import VideoStreamTrack

RESOLUTIONS = {
    "144p (256×144)":   (256, 144),
    "240p (426×240)":   (426, 240),
    "360p (640×360)":   (640, 360),
    "720p (1280×720)":  (1280, 720),
    "1080p (1920×1080)":(1920, 1080),
}

FALLBACK_CURSOR_FILES = {
    "arrow": "Aero Arrow.cur",
    "ibeam": "Aero iBeam.cur",
    "link": "Aero Link.cur",
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
        self._cursor_images: dict[str, Image.Image] = {}
        self._cursor_cache: dict[Tuple[str, Tuple[int, int]], Image.Image] = {}
        self._default_cursor_shape: Optional[str] = None
        self._system_cursor_handle: Optional[int] = None
        self._system_cursor_failed = False
        self._pointer_visible = False
        self._pointer_norm: Tuple[float, float] = (0.0, 0.0)
        self._remote_pointer_shape: Optional[str] = None
        self._local_pointer_visible = False
        self._local_pointer_norm: Tuple[float, float] = (0.0, 0.0)
        self._local_pointer_shape: Optional[str] = None
        self._pointer_lock = threading.Lock()
        self._last_capture_bbox: Optional[Tuple[int, int, int, int]] = None

        self._load_cursor()
        self._init_capture()

    def _load_cursor(self) -> None:
        self._cursor_images.clear()
        self._cursor_cache.clear()
        self._default_cursor_shape = None
        self._system_cursor_handle = None
        self._system_cursor_failed = False

        self._load_fallback_cursors()

        if sys.platform.startswith("win"):
            try:
                from ..ui.cursor_utils import get_system_cursor_pixmap

                pixmap, handle, shape = get_system_cursor_pixmap()
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Nem sikerült lekérni a rendszer kurzorát.")
                pixmap = None
                handle = None
                shape = None

            if handle is not None:
                self._system_cursor_handle = handle

            if pixmap and not pixmap.isNull():
                pil_cursor = self._convert_qpixmap_to_pil(pixmap)
                if pil_cursor:
                    self._cursor_images["system"] = self._sanitize_cursor_transparency(pil_cursor)
                    self._default_cursor_shape = "system"
            elif shape and shape in self._cursor_images and self._default_cursor_shape is None:
                self._default_cursor_shape = shape

        if self._default_cursor_shape is None:
            if "arrow" in self._cursor_images:
                self._default_cursor_shape = "arrow"
            elif self._cursor_images:
                self._default_cursor_shape = next(iter(self._cursor_images))

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

    def _load_fallback_cursors(self) -> None:
        media_dir = Path(__file__).resolve().parent

        for shape, filename in FALLBACK_CURSOR_FILES.items():
            candidate = media_dir / filename
            if not candidate.exists():
                continue
            try:
                loaded = Image.open(candidate).convert("RGBA")
            except Exception as exc:  # pragma: no cover - best effort load
                logger.debug("Nem sikerült betölteni a %s kurzort: %s", candidate, exc)
                continue
            self._cursor_images[shape] = self._sanitize_cursor_transparency(loaded)

        if "arrow" not in self._cursor_images:
            legacy_cursor = media_dir / "cursor.png"
            if legacy_cursor.exists():
                try:
                    loaded = Image.open(legacy_cursor).convert("RGBA")
                except Exception:
                    pass
                else:
                    self._cursor_images["arrow"] = self._sanitize_cursor_transparency(loaded)

    @staticmethod
    def _convert_qpixmap_to_pil(pixmap: QtGui.QPixmap) -> Optional[Image.Image]:
        if not pixmap or pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return None
        ptr = image.bits()
        ptr.setsize(image.width() * image.height() * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((image.height(), image.width(), 4))
        return Image.fromarray(arr, mode="RGBA")

    def _purge_cursor_cache_for(self, shape: str) -> None:
        to_remove = [key for key in self._cursor_cache if key[0] == shape]
        for key in to_remove:
            self._cursor_cache.pop(key, None)

    def _ensure_system_cursor(self, cursor_handle: Optional[int]) -> None:
        if not sys.platform.startswith("win"):
            return
        if cursor_handle is None:
            return
        if self._system_cursor_handle == cursor_handle:
            if "system" in self._cursor_images or self._system_cursor_failed:
                return

        try:
            from ..ui.cursor_utils import get_system_cursor_pixmap
        except Exception:  # pragma: no cover - defensive guard
            return

        pixmap, handle, _shape = get_system_cursor_pixmap()
        if handle is None:
            self._system_cursor_failed = True
            return

        self._system_cursor_handle = handle

        if pixmap and not pixmap.isNull():
            pil_cursor = self._convert_qpixmap_to_pil(pixmap)
            if pil_cursor:
                self._cursor_images["system"] = self._sanitize_cursor_transparency(pil_cursor)
                self._purge_cursor_cache_for("system")
                self._default_cursor_shape = "system"
                self._system_cursor_failed = False
                return

        self._system_cursor_failed = True

    def _resolve_cursor_shape(self, requested: Optional[str]) -> Optional[str]:
        if requested and requested in self._cursor_images:
            return requested
        if requested and requested != "arrow" and "arrow" in self._cursor_images:
            return "arrow"
        if "arrow" in self._cursor_images:
            return "arrow"
        if self._default_cursor_shape and self._default_cursor_shape in self._cursor_images:
            return self._default_cursor_shape
        if self._cursor_images:
            return next(iter(self._cursor_images))
        return None

    def _select_local_shape(self, requested: Optional[str]) -> Optional[str]:
        if "system" in self._cursor_images:
            return "system"
        return self._resolve_cursor_shape(requested)

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

    def _get_cursor_for_size(
        self, frame_size: Tuple[int, int], shape: Optional[str]
    ) -> Optional[Image.Image]:
        key = self._resolve_cursor_shape(shape)
        if not key:
            return None
        base = self._cursor_images.get(key)
        if base is None:
            return None
        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        if base.width <= 0 or base.height <= 0:
            return None
        target_width = min(base.width, max(CURSOR_MIN_SIZE, int(width * CURSOR_SCALE_RATIO)))
        if target_width <= 0:
            return None
        aspect = base.width / base.height
        target_height = min(base.height, max(CURSOR_MIN_SIZE, int(target_width / aspect)))
        if target_height <= 0:
            return None
        size = (target_width, target_height)
        cache_key = (key, size)
        cached = self._cursor_cache.get(cache_key)
        if cached is None:
            cached = base.resize(size, Image.LANCZOS)
            self._cursor_cache[cache_key] = cached
        return cached

    def _apply_pointer_overlay(self, img: Image.Image) -> Image.Image:
        with self._pointer_lock:
            pointer_states = []
            if self._local_pointer_visible:
                pointer_states.append((self._local_pointer_norm, self._local_pointer_shape))
            if self._pointer_visible:
                pointer_states.append((self._pointer_norm, self._remote_pointer_shape))

        if not pointer_states:
            return img

        width, height = img.size
        base = img.convert("RGBA")

        for (norm_x, norm_y), shape in pointer_states:
            cursor = self._get_cursor_for_size((width, height), shape)
            if cursor is None:
                continue
            cw, ch = cursor.size
            x = int(round(norm_x * width + CURSOR_OFFSET_X))
            y = int(round(norm_y * height + CURSOR_OFFSET_Y))
            x = max(0, min(x, max(0, width - cw)))
            y = max(0, min(y, max(0, height - ch)))
            base.paste(cursor, (x, y), cursor)

        return base

    def set_remote_pointer(self, norm_x: float, norm_y: float) -> None:
        with self._pointer_lock:
            self._pointer_visible = True
            self._pointer_norm = (
                float(max(0.0, min(1.0, norm_x))),
                float(max(0.0, min(1.0, norm_y))),
            )
            self._remote_pointer_shape = None

    def clear_remote_pointer(self) -> None:
        with self._pointer_lock:
            self._pointer_visible = False
            self._remote_pointer_shape = None

    def set_local_pointer(
        self,
        norm_x: float,
        norm_y: float,
        *,
        shape: Optional[str] = None,
        cursor_handle: Optional[int] = None,
    ) -> None:
        self._ensure_system_cursor(cursor_handle)
        with self._pointer_lock:
            self._local_pointer_visible = True
            self._local_pointer_norm = (
                float(max(0.0, min(1.0, norm_x))),
                float(max(0.0, min(1.0, norm_y))),
            )
            self._local_pointer_shape = self._select_local_shape(shape)

    def clear_local_pointer(self) -> None:
        with self._pointer_lock:
            self._local_pointer_visible = False
            self._local_pointer_shape = None

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
