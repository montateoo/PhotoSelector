"""
PhotoSelector — Photo viewer with one-key selection and batch copy.

  ← / →  or  A / D   navigate
  Space                select / deselect current photo
  Space (hold 2 s)     move current photo to "Scartate" folder
  Ctrl+Z               undo last discard (move back)
  I                    toggle info panel
  [ / ]                rotate left / right (saved to file automatically)
  +  /  -              zoom in / out
  0                    fit to window
  Ctrl+O               open folder
"""
from __future__ import annotations

import json
import sys
import os
import time
import shutil
import datetime
import platform
import logging
import hashlib
import traceback
import faulthandler
import urllib.request
import urllib.parse
import urllib.error
import tomllib
from pathlib import Path

from PIL import Image as PilImage, ImageOps as PilImageOps
from PIL.ExifTags import TAGS, GPSTAGS

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QScrollArea,
    QSizePolicy, QStatusBar, QFrame, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
    QProgressDialog, QProgressBar, QSlider,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QStyledItemDelegate, QListView, QStyle,
    QDialog, QLineEdit, QDialogButtonBox, QInputDialog, QMenu, QSpinBox,
)

import cv2 as _cv2
import numpy as _np
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QRect, QRectF, QPoint, QPointF,
                          QSize, QUrl, QTimer, QPropertyAnimation, pyqtProperty)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPainterPath, QFont, QBrush, QPen, QIcon,
    QKeySequence, QDesktopServices,
)

# ── Resource path (dev and PyInstaller --onefile) ─────────────────────────────

def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base) / name)


def _load_gh_token() -> str | None:
    """Reads secrets.toml (gitignored, bundled into the exe at build time)."""
    try:
        with open(resource_path("secrets.toml"), "rb") as f:
            data = tomllib.load(f)
        return data.get("github", {}).get("token") or None
    except Exception:
        return None


GH_TOKEN = _load_gh_token()

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tiff", ".tif", ".webp",
})
DEST_FOLDER    = "Selezionate"
DISCARD_FOLDER = "Scartate"
THUMB_W, THUMB_H = 132, 90

FILEMAIL_BASE  = "https://www.filemail.com"
# Filemail recommends chunks of 5–50 MB (for files >50 MB). Photos here are
# <10 MB, so a 25 MB chunk means every photo is sent in a single HTTP request —
# no server-side chunk assembly, no partial-file corruption risk.
FILEMAIL_CHUNK      = 25 * 1024 * 1024
FILEMAIL_FREE_LIMIT = 5 * 1024 * 1024 * 1024   # 5 GB free-tier cap
SETTINGS_PATH       = Path.home() / ".photoselector_settings.json"
WATERMARK_DIR       = Path.home() / ".photoselector_watermarks"


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    current = _load_settings()
    current.update(data)
    SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _fmt_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f} KB"
    return f"{n} B"


# ── Watermark helpers ─────────────────────────────────────────────────────────

def _load_watermarks() -> dict:
    WATERMARK_DIR.mkdir(exist_ok=True)
    return {
        "dark":  (WATERMARK_DIR / "dark.png")  if (WATERMARK_DIR / "dark.png").exists()  else None,
        "light": (WATERMARK_DIR / "light.png") if (WATERMARK_DIR / "light.png").exists() else None,
    }


def _pick_wm_path(img_bgr: '_np.ndarray', wms: dict) -> 'Path | None':
    """Auto-selects dark or light watermark based on luminance of the bottom-center region."""
    dark, light = wms.get("dark"), wms.get("light")
    if dark is None and light is None:
        return None
    if dark is None:
        return light
    if light is None:
        return dark
    h, w = img_bgr.shape[:2]
    region = img_bgr[int(h * 0.80):, int(w * 0.20):int(w * 0.80)]
    gray   = _cv2.cvtColor(region, _cv2.COLOR_BGR2GRAY)
    return dark if float(_np.mean(gray)) > 128 else light


def _composite_watermark(img_bgr: '_np.ndarray', wm_bgra: '_np.ndarray',
                          width_frac: float = 0.22, margin_frac: float = 0.03) -> '_np.ndarray':
    """Composites watermark centered at the bottom of img_bgr."""
    ih, iw   = img_bgr.shape[:2]
    wh0, ww0 = wm_bgra.shape[:2]
    wm_w = max(1, int(iw * width_frac))
    wm_h = max(1, int(wh0 * wm_w / max(1, ww0)))
    wm   = _cv2.resize(wm_bgra, (wm_w, wm_h), interpolation=_cv2.INTER_LANCZOS4)
    margin = max(4, int(ih * margin_frac))
    x  = (iw - wm_w) // 2
    y  = max(0, ih - wm_h - margin)
    ye = min(y + wm_h, ih)
    xe = min(x + wm_w, iw)
    ah = ye - y
    aw = xe - x
    out = img_bgr.astype(_np.float32)
    roi = out[y:ye, x:xe]
    wc  = wm[:ah, :aw]
    if wc.shape[2] == 4:
        a = wc[:, :, 3:4].astype(_np.float32) / 255.0
        c = wc[:, :, :3].astype(_np.float32)
    else:
        a = _np.ones((ah, aw, 1), dtype=_np.float32)
        c = wc.astype(_np.float32)
    roi[:] = c * a + roi * (1.0 - a)
    return out.astype(_np.uint8)


def _logo_xy(pos: str, iw: int, ih: int, lw: int, lh: int, margin: int) -> tuple:
    """Returns (x, y) top-left corner for logo placement given a position key."""
    v_part = pos.split("-")[0]                  # "top", "mid", "bot"
    h_part = pos.split("-")[1] if "-" in pos else "center"  # "left","center","right"
    x = {"left": margin, "center": (iw - lw) // 2, "right": iw - lw - margin}[h_part]
    y = {"top":  margin, "mid":    (ih - lh) // 2,  "bot":  ih - lh - margin}[v_part]
    return max(0, x), max(0, y)


def _composite_logo(img_bgr: '_np.ndarray', logo_bgra: '_np.ndarray',
                    pos: str, width_frac: float = 0.15,
                    margin_frac: float = 0.02) -> '_np.ndarray':
    """Composites logo onto img_bgr at the given position key."""
    ih, iw   = img_bgr.shape[:2]
    lh0, lw0 = logo_bgra.shape[:2]
    lw  = max(1, int(iw * width_frac))
    lh  = max(1, int(lh0 * lw / max(1, lw0)))
    logo = _cv2.resize(logo_bgra, (lw, lh), interpolation=_cv2.INTER_LANCZOS4)
    margin = max(4, int(min(iw, ih) * margin_frac))
    x, y = _logo_xy(pos, iw, ih, lw, lh, margin)
    ye = min(y + lh, ih);  xe = min(x + lw, iw)
    ah = ye - y;            aw = xe - x
    out = img_bgr.astype(_np.float32)
    roi = out[y:ye, x:xe]
    wc  = logo[:ah, :aw]
    if wc.shape[2] == 4:
        a = wc[:, :, 3:4].astype(_np.float32) / 255.0
        c = wc[:, :, :3].astype(_np.float32)
    else:
        a = _np.ones((ah, aw, 1), dtype=_np.float32)
        c = wc.astype(_np.float32)
    roi[:] = c * a + roi * (1.0 - a)
    return out.astype(_np.uint8)


# ── Crash logging + GitHub auto-reporting ─────────────────────────────────────

GH_OWNER = "montateoo"
GH_REPO  = "PhotoSelector"

_APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PhotoSelector"
_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CRASH_FAULT_PATH = _APP_DATA_DIR / "crash_native.log"
_APP_LOG_PATH     = _APP_DATA_DIR / "app.log"
_STATE_PATH       = _APP_DATA_DIR / "session_state.json"
_THROTTLE_PATH    = _APP_DATA_DIR / "reported_signatures.json"

_fault_fp = open(_CRASH_FAULT_PATH, "a", encoding="utf-8")
faulthandler.enable(_fault_fp)

class _FlushHandler(logging.FileHandler):
    """FileHandler that flushes after every record so entries survive hard crashes."""
    def emit(self, record):
        super().emit(record)
        self.flush()

_log_handler = _FlushHandler(str(_APP_LOG_PATH), encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(handlers=[_log_handler], level=logging.INFO)
log = logging.getLogger("PhotoSelector")

_session_state = {"action": "starting", "detail": "", "photos": 0}


def _set_action(action: str, detail: str = "", photos: int | None = None) -> None:
    """Records a breadcrumb of what the app was doing, for crash context."""
    _session_state["action"] = action
    _session_state["detail"] = detail
    if photos is not None:
        _session_state["photos"] = photos
    try:
        payload = json.dumps({**_session_state, "clean_exit": False, "ts": time.time()})
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(_STATE_PATH)   # atomic on same filesystem
    except Exception:
        pass


def _mark_clean_exit() -> None:
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        data["clean_exit"] = True
        _STATE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _system_info_text() -> str:
    return (
        f"OS: {platform.platform()}\n"
        f"Python: {platform.python_version()}\n"
        f"OpenCV: {_cv2.__version__}\n"
    )


def _already_reported_recently(signature: str) -> bool:
    """Avoid spamming the issue tracker if the same crash repeats in a loop."""
    try:
        data = json.loads(_THROTTLE_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    last = data.get(signature)
    recent = bool(last and (time.time() - last) < 24 * 3600)
    data[signature] = time.time()
    try:
        _THROTTLE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return recent


def _report_crash_to_github(title: str, body: str, signature: str) -> None:
    log.error(f"{title}\n{body}")
    if not GH_TOKEN:
        log.warning("No GitHub token bundled — crash logged locally only")
        return
    if _already_reported_recently(signature):
        log.info("Same crash signature reported in the last 24h — skipping upload")
        return
    try:
        url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/issues"
        payload = json.dumps({"title": title, "body": body, "labels": ["crash"]}).encode()
        req = urllib.request.Request(url, data=payload, method="POST", headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "PhotoSelector-CrashReporter",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            log.info(f"GitHub issue created (status={resp.status})")
    except Exception as e:
        log.error(f"Failed to upload crash report to GitHub: {e}")


def _format_crash_report(exc_type, exc_value, exc_tb) -> tuple[str, str]:
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    title = f"[Crash] {exc_type.__name__}: {str(exc_value)[:80]}"
    body = (
        "### Crash automatico\n\n"
        f"**Quando:** {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"**Azione in corso:** {_session_state.get('action')} — {_session_state.get('detail')}\n"
        f"**Foto caricate:** {_session_state.get('photos')}\n\n"
        f"**Sistema:**\n```\n{_system_info_text()}```\n\n"
        f"**Traceback:**\n```\n{tb_text}```"
    )
    return title, body


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    title, body = _format_crash_report(exc_type, exc_value, exc_tb)
    signature = hashlib.sha1(f"{exc_type.__name__}:{_session_state.get('action')}"
                             .encode()).hexdigest()[:12]
    _report_crash_to_github(title, body, signature)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


def _check_previous_session_crash() -> None:
    """Detects a process that died without a clean exit (native crash / OOM kill),
    which never reaches _excepthook, and reports it retroactively on next launch."""
    try:
        if not _STATE_PATH.exists():
            return
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if data.get("clean_exit", True):
            return
        title = "[Crash] Uscita anomala (possibile crash nativo o memoria insufficiente)"
        body = (
            "### Uscita anomala rilevata al lancio successivo\n\n"
            "L'app non si è chiusa correttamente nella sessione precedente "
            "(nessuna eccezione Python intercettata — probabile crash nativo o OOM).\n\n"
            f"**Ultima azione registrata:** {data.get('action')} — {data.get('detail')}\n"
            f"**Foto caricate:** {data.get('photos')}\n"
            f"**Timestamp ultima azione:** "
            f"{datetime.datetime.fromtimestamp(data.get('ts', 0)).isoformat(timespec='seconds')}\n\n"
            f"**Sistema:**\n```\n{_system_info_text()}```"
        )
        signature = hashlib.sha1(f"abnormal_exit:{data.get('action')}".encode()).hexdigest()[:12]
        _report_crash_to_github(title, body, signature)
    except Exception as e:
        log.error(f"Failed to check previous session state: {e}")

# ── Stylesheets ───────────────────────────────────────────────────────────────

BASE_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a1a;
    color: #f0f0f0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QScrollArea { border: none; background: #222; }
QScrollBar:horizontal {
    height: 6px; background: #2a2a2a; border: none;
}
QScrollBar::handle:horizontal {
    background: #555; border-radius: 3px; min-width: 30px;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar:vertical {
    width: 6px; background: #2a2a2a; border: none;
}
QScrollBar::handle:vertical {
    background: #555; border-radius: 3px; min-height: 30px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #222; color: #888; font-size: 12px; }
QProgressDialog { background: #2a2a2a; }
QProgressDialog QLabel { color: #f0f0f0; }
QMessageBox { background: #2a2a2a; }
QMessageBox QLabel { color: #f0f0f0; }
QPushButton {
    background: #2d2d2d; color: #f0f0f0;
    border: none; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover   { background: #3d3d3d; }
QPushButton:pressed { background: #1e1e1e; }
QPushButton:disabled { background: #252525; color: #555; }
QProgressBar {
    border: none; background: #2a2a2a;
    text-align: center; color: transparent;
}
QProgressBar::chunk { background: #0078d4; border-radius: 0px; }
"""

ACCENT_STYLE = """
QPushButton {
    background: #0078d4; color: #fff;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold;
}
QPushButton:hover   { background: #106ebe; }
QPushButton:pressed { background: #005a9e; }
QPushButton:disabled { background: #1a3a5c; color: #666; }
"""

SELECT_ACTIVE_STYLE = """
QPushButton {
    background: #00a060; color: #fff;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold;
}
QPushButton:hover   { background: #008850; }
QPushButton:pressed { background: #006640; }
"""

INFO_ACTIVE_STYLE = """
QPushButton {
    background: #3a3a3a; color: #fff;
    border: 1px solid #555; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover   { background: #4a4a4a; }
QPushButton:pressed { background: #2a2a2a; }
"""

NAV_ARROW_STYLE = """
QPushButton {
    background: rgba(30,30,30,180); color: #fff;
    border: none; border-radius: 6px;
    font-size: 26px; font-weight: bold;
}
QPushButton:hover   { background: rgba(60,60,60,210); }
QPushButton:pressed { background: rgba(0,120,212,200); }
QPushButton:disabled { color: #333; background: transparent; }
"""

SHARE_STYLE = """
QPushButton {
    background: #6b3fa0; color: #fff;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold;
}
QPushButton:hover   { background: #7d50b8; }
QPushButton:pressed { background: #4e2e7a; }
QPushButton:disabled { background: #2a1e40; color: #666; }
"""

WATERMARK_ACTIVE_STYLE = """
QPushButton {
    background: #c87800; color: #fff;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold;
}
QPushButton:hover   { background: #b06800; }
QPushButton:pressed { background: #9a5a00; }
"""

LOGO_ACTIVE_STYLE = """
QPushButton {
    background: #007b8a; color: #fff;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold;
}
QPushButton:hover   { background: #006778; }
QPushButton:pressed { background: #005566; }
"""

_LOGO_POS_GRID = [
    ("top-left",   "↖"), ("top-center",   "↑"), ("top-right",   "↗"),
    ("mid-left",   "←"), ("mid-center",   "⊕"), ("mid-right",   "→"),
    ("bot-left",   "↙"), ("bot-center",   "↓"), ("bot-right",   "↘"),
]

_LOGO_POS_STYLE = (
    "QPushButton {"
    " background:#2a2a2a; color:#bbb; border:1px solid #404040;"
    " border-radius:4px; font-size:18px; font-weight:bold;"
    "}"
    "QPushButton:hover  { background:#3a3a3a; color:#eee; }"
    "QPushButton:checked {"
    " background:#007b8a; color:#fff; border-color:#007b8a;"
    "}"
)

# ── Orientation-aware image loading ───────────────────────────────────────────

def _load_oriented_pixmap(path: Path, max_size: tuple[int, int] | None = None) -> QPixmap:
    """Loads an image applying its EXIF orientation tag (QPixmap alone ignores it).

    When max_size is given, uses PIL's JPEG draft mode to downscale during
    decode instead of after — this keeps thumbnail generation for large photo
    batches from spiking memory with full-resolution decodes.
    """
    try:
        with PilImage.open(str(path)) as img:
            if max_size is not None and img.format == "JPEG":
                img.draft("RGB", max_size)
            img = PilImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            if max_size is not None:
                img.thumbnail(max_size, PilImage.LANCZOS)
            w, h = img.size
            data = img.tobytes("raw", img.mode)
            fmt  = QImage.Format_RGBA8888 if img.mode == "RGBA" else QImage.Format_RGB888
            stride = (4 if img.mode == "RGBA" else 3) * w
            qimg = QImage(data, w, h, stride, fmt)
            return QPixmap.fromImage(qimg.copy())
    except Exception as e:
        log.warning(f"Orientation-aware load failed for {path.name}: {e}")
        return QPixmap(str(path))


# ── EXIF utilities ────────────────────────────────────────────────────────────

def _rational(v) -> float:
    """Convert IFRational / (num, den) tuple / plain number to float."""
    if isinstance(v, tuple) and len(v) == 2:
        return v[0] / v[1] if v[1] else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0

def _dms_to_deg(dms) -> float:
    return _rational(dms[0]) + _rational(dms[1]) / 60 + _rational(dms[2]) / 3600

def read_photo_info(path: Path) -> dict:
    info: dict = {
        "filename": path.name,
        "filesize": path.stat().st_size,
        "width": 0, "height": 0,
        "date_taken": None,
        "camera_make": "", "camera_model": "",
        "lens": "", "focal_length": "",
        "aperture": "", "shutter_speed": "", "iso": "",
        "gps_lat": None, "gps_lon": None,
    }
    try:
        with PilImage.open(str(path)) as img:
            info["width"]  = img.width
            info["height"] = img.height
            exif = img.getexif()
            if not exif:
                return info

            # DateTimeOriginal, LensModel, FocalLength, FNumber, ExposureTime
            # and ISO live in the Exif SubIFD (0x8769), not the top-level IFD0
            # that exif.items() alone returns — merge both so nothing is missed.
            merged = {**dict(exif), **exif.get_ifd(0x8769)}

            for tag_id, val in merged.items():
                tag = TAGS.get(tag_id, "")
                if tag == "DateTimeOriginal":
                    try:
                        info["date_taken"] = datetime.datetime.strptime(
                            str(val), "%Y:%m:%d %H:%M:%S"
                        )
                    except ValueError:
                        pass
                elif tag == "Make":
                    info["camera_make"] = str(val).strip()
                elif tag == "Model":
                    info["camera_model"] = str(val).strip()
                elif tag == "LensModel":
                    info["lens"] = str(val).strip()
                elif tag == "FocalLength":
                    f = _rational(val)
                    if f:
                        info["focal_length"] = f"{f:.0f} mm"
                elif tag == "FNumber":
                    f = _rational(val)
                    if f:
                        info["aperture"] = f"f/{f:.1f}"
                elif tag == "ExposureTime":
                    t = _rational(val)
                    if t > 0:
                        info["shutter_speed"] = (
                            f"1/{round(1/t)}s" if t < 1 else f"{t:.1f}s"
                        )
                elif tag == "ISOSpeedRatings":
                    iso = val[0] if isinstance(val, (list, tuple)) else val
                    info["iso"] = f"ISO {iso}"

            # GPS
            gps_ifd = exif.get_ifd(0x8825)
            if gps_ifd:
                gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                try:
                    lat = _dms_to_deg(gps["GPSLatitude"])
                    if gps.get("GPSLatitudeRef", "N") != "N":
                        lat = -lat
                    lon = _dms_to_deg(gps["GPSLongitude"])
                    if gps.get("GPSLongitudeRef", "E") != "E":
                        lon = -lon
                    info["gps_lat"] = round(lat, 6)
                    info["gps_lon"] = round(lon, 6)
                except (KeyError, TypeError, IndexError, ZeroDivisionError):
                    pass
    except Exception:
        pass
    return info

# ── Background workers ────────────────────────────────────────────────────────

class ThumbnailLoader(QThread):
    ready    = pyqtSignal(int, QPixmap)
    progress = pyqtSignal(int)          # number of thumbs processed so far

    def __init__(self, photos: list[Path]):
        super().__init__()
        self.photos   = photos
        self._active  = True

    def run(self):
        import gc
        n = len(self.photos)
        for i, path in enumerate(self.photos):
            if not self._active:
                break
            if i % 10 == 0:
                _set_action("generating_thumbnails", f"{i}/{n}", n)
                gc.collect()          # release accumulated PIL/Qt memory
                if i > 0:
                    self.msleep(5)    # let the main thread drain queued signals
            try:
                px = _load_oriented_pixmap(path, max_size=(THUMB_W * 2, THUMB_H * 2))
                if not px.isNull():
                    thumb = px.scaled(THUMB_W, THUMB_H,
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
                    self.ready.emit(i, thumb)
                del px
            except MemoryError:
                log.error(f"Out of memory generating thumbnail for {path.name} — skipping")
            except Exception as e:
                log.error(f"Thumbnail generation failed for {path.name}: {e}")
            self.progress.emit(i + 1)

    def stop(self):
        self._active = False


class ImageLoader(QThread):
    ready = pyqtSignal(int, QPixmap)

    def __init__(self, index: int, path: Path):
        super().__init__()
        self.index = index
        self.path  = path

    def run(self):
        try:
            self.ready.emit(self.index, _load_oriented_pixmap(self.path))
        except Exception as e:
            log.error(f"Image load failed for {self.path.name}: {e}")
            self.ready.emit(self.index, QPixmap())


class ExifLoader(QThread):
    ready = pyqtSignal(int, dict)

    def __init__(self, index: int, path: Path):
        super().__init__()
        self.index = index
        self.path  = path

    def run(self):
        self.ready.emit(self.index, read_photo_info(self.path))


class RotateWorker(QThread):
    """Rotates an image file in-place and preserves EXIF (except orientation)."""
    done  = pyqtSignal(int)   # emits the photo index on success
    error = pyqtSignal(str)

    def __init__(self, index: int, path: Path, clockwise: bool):
        super().__init__()
        self.index     = index
        self.path      = path
        self.clockwise = clockwise

    def run(self):
        try:
            with PilImage.open(str(self.path)) as img:
                exif = img.getexif()

                # Apply any existing EXIF orientation FIRST so our rotation is
                # relative to what the user actually sees on screen. Must run
                # before popping the tag below — exif_transpose reads it from
                # the same cached Exif object getexif() returned.
                img = PilImageOps.exif_transpose(img)

                # Drop the orientation tag now that it's baked into the pixels
                exif.pop(274, None)          # 274 = Orientation
                exif_bytes = exif.tobytes()

                # PIL rotates counter-clockwise, so clockwise → -90
                rotated = img.rotate(-90 if self.clockwise else 90, expand=True)

                suffix = self.path.suffix.lower()
                if suffix in (".jpg", ".jpeg"):
                    rotated.save(str(self.path), format="JPEG",
                                 quality=95, subsampling=0, exif=exif_bytes)
                elif suffix == ".png":
                    rotated.save(str(self.path), format="PNG")
                else:
                    rotated.save(str(self.path))

            self.done.emit(self.index)
        except Exception as e:
            self.error.emit(str(e))


class GeocoderThread(QThread):
    """Reverse-geocodes lat/lon via Nominatim (no API key required)."""
    ready = pyqtSignal(str)

    def __init__(self, lat: float, lon: float):
        super().__init__()
        self.lat = lat
        self.lon = lon

    def run(self):
        try:
            url = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"?lat={self.lat}&lon={self.lon}&format=json&zoom=10"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "PhotoSelector/1.0"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
            addr = data.get("address", {})
            parts: list[str] = []
            for key in ("city", "town", "village", "municipality",
                        "county", "state", "country"):
                v = addr.get(key, "")
                if v and v not in parts:
                    parts.append(v)
                if len(parts) >= 3:
                    break
            self.ready.emit(", ".join(parts) or data.get("display_name", "")[:80])
        except Exception:
            self.ready.emit("")


class FilemailUploader(QThread):
    """Uploads a list of files to Filemail and emits the download URL."""
    progress = pyqtSignal(int, int)   # (files_done, files_total)
    done     = pyqtSignal(str)        # shareable download URL
    warning  = pyqtSignal(str)        # non-fatal: some files failed but others succeeded
    error    = pyqtSignal(str)

    def __init__(self, files: list, from_email: str, api_key: str = ""):
        super().__init__()
        self._files      = list(files)
        self._from_email = from_email
        self._api_key    = api_key
        self._stopped    = False

    def stop(self):
        self._stopped = True

    def run(self):
        failed: list[str] = []
        uploaded = 0
        try:
            # 1. Initialize transfer
            params: dict = {
                "from":         self._from_email,
                "subject":      "Foto selezionate",
                "days":         "7",
                "downloads":    "0",
                "notify":       "false",
                "confirmation": "false",
                "compress":     "false",
            }
            if self._api_key:
                params["apikey"] = self._api_key

            init_url = (
                f"{FILEMAIL_BASE}/api/transfer/initialize?"
                + urllib.parse.urlencode(params)
            )
            req = urllib.request.Request(
                init_url, headers={"User-Agent": "PhotoSelector/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            if data.get("response") not in ("ok", None):
                raise RuntimeError(
                    data.get("errormessage", "Errore inizializzazione trasferimento")
                )

            t    = data.get("transfer", data)
            tid  = t["transferid"]
            tkey = t["transferkey"]
            turl = t["transferurl"]

            # 2. Upload files one by one — per-file errors are isolated so a
            #    single bad file cannot abandon the whole transfer on Filemail's
            #    server (which would leave partial bytes that look corrupted).
            #    Each file gets FILE_ATTEMPTS full-file retries; chunk-level
            #    retries happen inside _upload_file/_send_chunk_with_retry.
            FILE_ATTEMPTS = 3
            for i, path in enumerate(self._files):
                if self._stopped:
                    break
                p = Path(path)
                last_err: Exception | None = None
                for file_attempt in range(1, FILE_ATTEMPTS + 1):
                    if self._stopped:
                        break
                    try:
                        self._upload_file(p, turl, tid, tkey)
                        uploaded += 1
                        last_err = None
                        break
                    except Exception as file_err:
                        last_err = file_err
                        if file_attempt < FILE_ATTEMPTS:
                            log.warning(
                                f"File attempt {file_attempt}/{FILE_ATTEMPTS} failed "
                                f"for {p.name}: {file_err} — retrying whole file"
                            )
                            time.sleep(2 * file_attempt)
                if last_err is not None:
                    log.error(
                        f"Upload permanently failed for {p.name} "
                        f"after {FILE_ATTEMPTS} attempts: {last_err}"
                    )
                    failed.append(p.name)
                self.progress.emit(i + 1, len(self._files))

            # If the user cancelled, don't seal the transfer.
            if self._stopped:
                return

            if uploaded == 0:
                raise RuntimeError("Nessun file è stato caricato con successo.")

            # 3. Complete transfer — always called so Filemail properly seals
            #    it, even when some files errored above.
            cparams: dict = {"transferid": tid, "transferkey": tkey}
            if self._api_key:
                cparams["apikey"] = self._api_key
            complete_url = (
                f"{FILEMAIL_BASE}/api/transfer/complete?"
                + urllib.parse.urlencode(cparams)
            )
            req2 = urllib.request.Request(
                complete_url, headers={"User-Agent": "PhotoSelector/1.0"}
            )
            with urllib.request.urlopen(req2, timeout=30) as resp2:
                data2 = json.loads(resp2.read())

            t2     = data2.get("transfer", data2)
            dl_url = t2.get("downloadurl") or t2.get("url") or ""
            if not dl_url:
                raise RuntimeError(
                    "Trasferimento completato ma link non trovato nella risposta"
                )
            self.done.emit(dl_url)

            if failed:
                self.warning.emit(
                    f"{len(failed)} foto non caricate (riprovare separatamente):\n"
                    + "\n".join(f"• {n}" for n in failed)
                )

        except Exception as exc:
            self.error.emit(str(exc))

    def _upload_file(self, path: Path, turl: str, tid: str, tkey: str):
        # Quick sanity check before consuming any upload quota: verify the file
        # is a readable image. A locally-corrupt file would burn all retry
        # attempts against the server for no reason.
        try:
            with PilImage.open(str(path)) as _img:
                _img.verify()
        except Exception as e:
            raise RuntimeError(f"Il file è danneggiato o non è un'immagine valida: {e}")

        size = path.stat().st_size
        with open(path, "rb") as fh:
            offset = 0
            while offset < size:
                if self._stopped:
                    return
                chunk = fh.read(FILEMAIL_CHUNK)
                if not chunk:
                    break
                qs = urllib.parse.urlencode({
                    "transferid":  tid,
                    "transferkey": tkey,
                    "chunkoffset": offset,
                    "totalsize":   size,
                    "thefilename": path.name,
                })
                headers = {
                    "Content-Type":   "application/octet-stream",
                    "Content-Length": str(len(chunk)),
                    "User-Agent":     "PhotoSelector/1.0",
                }
                self._send_chunk_with_retry(
                    f"{turl}?{qs}", chunk, headers, path.name, offset
                )
                offset += len(chunk)

        # Defensive check: a silently-truncated upload must not be reported as success
        if offset != size:
            raise RuntimeError(
                f"Caricamento incompleto per {path.name}: "
                f"inviati {offset} byte su {size}"
            )

    def _send_chunk_with_retry(
        self,
        url: str,
        chunk: bytes,
        headers: dict,
        filename: str,
        offset: int,
        attempts: int = 5,
    ):
        """Uploads one chunk, rebuilding the Request on every attempt.

        urllib.request.Request can accumulate internal state (added headers, handler
        references) after a failed urlopen call, causing subsequent retries to send a
        malformed request. Building a fresh Request each time avoids this.
        """
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            if self._stopped:
                return
            try:
                req = urllib.request.Request(
                    url, data=chunk, method="POST", headers=headers
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    body = resp.read()
                try:
                    parsed = json.loads(body) if body else None
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("response") not in (None, "ok"):
                    raise RuntimeError(parsed.get("errormessage", "errore sconosciuto"))
                return  # success
            except Exception as e:
                last_err = e
                if attempt < attempts:
                    time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s, 8s back-off
        raise RuntimeError(
            f"Caricamento del chunk fallito per {filename} (offset {offset}) "
            f"dopo {attempts} tentativi: {last_err}"
        )


# ── Filmstrip delegate (paints each thumbnail cell) ───────────────────────────

class ThumbDelegate(QStyledItemDelegate):
    def __init__(self, selected: set[int], parent=None):
        super().__init__(parent)
        self._selected = selected

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        row  = index.row()
        rect = option.rect

        painter.fillRect(rect, QColor("#2a2a2a"))

        # Thumbnail image
        icon = index.data(Qt.DecorationRole)
        if icon and not icon.isNull():
            px = icon.pixmap(THUMB_W, THUMB_H)
            x  = rect.x() + (rect.width()  - px.width())  // 2
            y  = rect.y() + (rect.height() - px.height()) // 2
            painter.drawPixmap(x, y, px)

        is_current = bool(option.state & QStyle.State_Selected)
        is_sel     = row in self._selected

        # Border: blue=current, green=selected
        if is_current or is_sel:
            color = QColor("#0078d4") if is_current else QColor("#00c878")
            pen = QPen(color, 3)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect.adjusted(2, 2, -2, -2))

        # Green checkmark badge for selected photos
        if is_sel:
            cx, cy, r = rect.right() - 15, rect.top() + 15, 11
            painter.setBrush(QBrush(QColor("#00c878")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
            pen = QPen(Qt.white, 2.2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(cx - 5, cy, cx - 1, cy + 4)
            painter.drawLine(cx - 1, cy + 4, cx + 5, cy - 3)

        painter.restore()

    def sizeHint(self, option, index):
        return QSize(THUMB_W + 8, THUMB_H + 8)

# ── Film strip ────────────────────────────────────────────────────────────────

class FilmStrip(QListWidget):
    navigate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._selected: set[int] = set()
        self.setItemDelegate(ThumbDelegate(self._selected, self))

        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setUniformItemSizes(True)      # only renders visible items
        self.setFixedHeight(THUMB_H + 28)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setIconSize(QSize(THUMB_W, THUMB_H))
        self.setMovement(QListView.Static)
        self.setSpacing(4)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setStyleSheet("""
            QListWidget {
                background: #222; border: none;
                padding: 4px 8px; outline: none;
            }
            QListWidget::item          { background: transparent; border: none; }
            QListWidget::item:selected { background: transparent; border: none; }
        """)
        self.currentRowChanged.connect(self.navigate.emit)

    def populate(self, count: int):
        self._selected.clear()
        self.clear()
        for _ in range(count):
            item = QListWidgetItem()
            item.setSizeHint(QSize(THUMB_W + 8, THUMB_H + 8))
            self.addItem(item)

    def set_thumbnail(self, index: int, px: QPixmap):
        item = self.item(index)
        if item:
            item.setIcon(QIcon(px))

    def set_current(self, index: int):
        # blockSignals prevents the currentRowChanged → navigate → set_current loop
        self.blockSignals(True)
        self.setCurrentRow(index)
        self.blockSignals(False)
        item = self.item(index)
        if item:
            self.scrollToItem(item, QAbstractItemView.EnsureVisible)

    def set_selected(self, index: int, v: bool):
        if v:
            self._selected.add(index)
        else:
            self._selected.discard(index)
        item = self.item(index)
        if item:
            self.update(self.indexFromItem(item))

    def reindex_selected(self, mapper):
        """Recompute indices in-place (mapper: old_index -> new_index, or None to drop).

        Must mutate self._selected rather than reassign it — ThumbDelegate was
        handed a direct reference to this set at construction time, so a fresh
        set here orphans the delegate from future updates (it keeps painting
        stale state forever, even though FilmStrip's own set is correct).
        """
        new_indices = {mapper(i) for i in self._selected}
        new_indices.discard(None)
        self._selected.clear()
        self._selected.update(new_indices)
        self.viewport().update()

# ── Crop canvas (interactive handles + pan + zoom) ────────────────────────────

_HANDLE_R = 6   # handle radius in widget pixels

_ASPECT_PRESETS: list[tuple[str, float | None]] = [
    ("Libero",    None),
    ("Originale", -1.0),
    ("1:1",       1.0),
    ("9:16",      9 / 16),
    ("16:9",      16 / 9),
    ("4:5",       4 / 5),
    ("5:4",       5 / 4),
    ("3:4",       3 / 4),
    ("4:3",       4 / 3),
    ("2:3",       2 / 3),
    ("3:2",       3 / 2),
    ("5:7",       5 / 7),
    ("7:5",       7 / 5),
    ("1:2",       1 / 2),
    ("2:1",       2 / 1),
]

_PRESET_BTN_STYLE = (
    "QPushButton {"
    " background:#2a2a2a; color:#bbb; border:1px solid #404040;"
    " border-radius:4px; padding:2px 8px; font-size:11px;"
    "}"
    "QPushButton:hover  { background:#3a3a3a; color:#eee; }"
    "QPushButton:checked {"
    " background:#0078d4; color:#fff; border-color:#0078d4; font-weight:bold;"
    "}"
)


class _CropCanvas(QWidget):
    """Crop widget where the frame lives in screen-space (widget coords).

    • Drag a handle  → resize the crop frame (frame stays in screen)
    • Drag inside    → move the crop frame
    • Drag outside   → pan the image behind the fixed frame
    • Mouse wheel    → zoom image around cursor (frame stays put)
    """

    def __init__(self, pixmap: QPixmap, orig_w: int, orig_h: int, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._orig_w = orig_w
        self._orig_h = orig_h
        self._aspect: float | None = None

        # View state: zoom + what image coord sits at the widget centre
        self._view_scale = 1.0
        self._center_img = QPointF(orig_w / 2.0, orig_h / 2.0)

        # Crop frame in WIDGET pixels — stable on screen while image moves
        self._crop_w: QRectF = QRectF()
        self._crop_init      = False    # set on first layout/paint

        # Drag state
        self._drag_mode       = ""
        self._drag_start_pos  = QPointF()
        self._drag_start_crop = QRectF()
        self._drag_start_ctr  = QPointF()

        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #111;")

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_crop(self):
        """Align crop frame to the full image at fit-to-window zoom."""
        ww, wh = float(self.width()), float(self.height())
        if ww == 0 or wh == 0:
            return
        self._view_scale = 1.0
        self._center_img = QPointF(self._orig_w / 2.0, self._orig_h / 2.0)
        self._crop_w     = QRectF(self._draw_rect())
        self._crop_init  = True
        if self._aspect is not None and self._aspect > 0:
            self._apply_aspect()

    # ── aspect ratio ──────────────────────────────────────────────────────────

    def set_aspect(self, aspect: float | None):
        self._aspect = aspect
        if self._crop_init:
            if aspect is not None and aspect > 0:
                self._apply_aspect()
            self.update()

    def _apply_aspect(self):
        """Resize crop frame to fill the image area at the current aspect ratio."""
        asp = self._aspect
        if not asp or asp <= 0:
            return
        dr = self._draw_rect()
        dw, dh = dr.width(), dr.height()
        if dw <= 0 or dh <= 0:
            return
        if dw / dh > asp:
            ch = dh;  cw = ch * asp
        else:
            cw = dw;  ch = cw / asp
        cx, cy = dr.center().x(), dr.center().y()
        self._crop_w = QRectF(cx - cw / 2, cy - ch / 2, cw, ch)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _display_scale(self) -> float:
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if not (pw and ph and ww and wh):
            return 1.0
        return min(ww / pw, wh / ph) * self._view_scale

    def _draw_rect(self) -> QRectF:
        ds = self._display_scale()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ox = self.width()  / 2 - self._center_img.x() * ds
        oy = self.height() / 2 - self._center_img.y() * ds
        return QRectF(ox, oy, pw * ds, ph * ds)

    def _widget_to_img(self, pt: QPointF) -> QPointF:
        dr = self._draw_rect()
        if not (dr.width() and dr.height()):
            return QPointF()
        return QPointF(
            (pt.x() - dr.x()) * self._orig_w / dr.width(),
            (pt.y() - dr.y()) * self._orig_h / dr.height(),
        )

    # ── handle helpers ────────────────────────────────────────────────────────

    def _handle_points(self) -> dict[str, QPointF]:
        r  = self._crop_w
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": r.topLeft(),     "t":  QPointF(cx, r.top()),
            "tr": r.topRight(),    "r":  QPointF(r.right(), cy),
            "br": r.bottomRight(), "b":  QPointF(cx, r.bottom()),
            "bl": r.bottomLeft(),  "l":  QPointF(r.left(), cy),
        }

    def _hit_handle(self, pos: QPointF) -> str:
        thresh = _HANDLE_R * 2.5
        for name, pt in self._handle_points().items():
            d = pos - pt
            if (d.x() ** 2 + d.y() ** 2) ** 0.5 <= thresh:
                return name
        return ""

    @staticmethod
    def _cursor_for(mode: str) -> Qt.CursorShape:
        return {
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            "t":  Qt.SizeVerCursor,   "b":  Qt.SizeVerCursor,
            "l":  Qt.SizeHorCursor,   "r":  Qt.SizeHorCursor,
            "move": Qt.SizeAllCursor, "pan": Qt.OpenHandCursor,
        }.get(mode, Qt.ArrowCursor)

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        if not self._crop_init:
            self._init_crop()

        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)

        p.drawPixmap(self._draw_rect().toRect(), self._pixmap)

        cr  = self._crop_w
        dim = QColor(0, 0, 0, 155)
        w, h = float(self.width()), float(self.height())
        p.fillRect(QRectF(0,          0,          w,               cr.top()),        dim)
        p.fillRect(QRectF(0,          cr.bottom(), w,               h - cr.bottom()), dim)
        p.fillRect(QRectF(0,          cr.top(),    cr.left(),        cr.height()),     dim)
        p.fillRect(QRectF(cr.right(), cr.top(),    w - cr.right(),   cr.height()),     dim)

        # Crop border
        p.setPen(QPen(Qt.white, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRect(cr)

        # Rule-of-thirds grid (always shown, clearly visible)
        p.setPen(QPen(QColor(255, 255, 255, 170), 1.5))
        for frac in (1 / 3, 2 / 3):
            p.drawLine(QPointF(cr.left() + cr.width() * frac, cr.top()),
                       QPointF(cr.left() + cr.width() * frac, cr.bottom()))
            p.drawLine(QPointF(cr.left(),  cr.top() + cr.height() * frac),
                       QPointF(cr.right(), cr.top() + cr.height() * frac))

        # Handles
        p.setPen(QPen(Qt.white, 1.5))
        p.setBrush(QColor(255, 255, 255, 220))
        for pt in self._handle_points().values():
            p.drawEllipse(pt, float(_HANDLE_R), float(_HANDLE_R))

        p.end()

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        pos    = QPointF(e.pos())
        handle = self._hit_handle(pos)
        if handle:
            self._drag_mode = handle
        elif self._crop_w.contains(pos):
            self._drag_mode = "move"
        else:
            self._drag_mode = "pan"
        self._drag_start_pos  = pos
        self._drag_start_crop = QRectF(self._crop_w)
        self._drag_start_ctr  = QPointF(self._center_img)
        self.setCursor(self._cursor_for(self._drag_mode))

    def mouseMoveEvent(self, e):
        pos = QPointF(e.pos())

        if not (e.buttons() & Qt.LeftButton):
            handle = self._hit_handle(pos)
            if handle:
                self.setCursor(self._cursor_for(handle))
            elif self._crop_w.contains(pos):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
            return

        delta = pos - self._drag_start_pos
        mode  = self._drag_mode

        if mode == "pan":
            ds = self._display_scale()
            if ds:
                self._center_img = QPointF(
                    self._drag_start_ctr.x() - delta.x() / ds,
                    self._drag_start_ctr.y() - delta.y() / ds,
                )
            self.update()
            return

        # Resize or move crop frame — all in widget pixels; no image scaling
        c = QRectF(self._drag_start_crop)
        if mode == "move":
            c.translate(delta)
        else:
            if "l" in mode: c.setLeft(c.left()     + delta.x())
            if "r" in mode: c.setRight(c.right()   + delta.x())
            if "t" in mode: c.setTop(c.top()       + delta.y())
            if "b" in mode: c.setBottom(c.bottom() + delta.y())
            c = c.normalized()

            asp = self._aspect
            if asp is not None and asp > 0:
                if mode in ("t", "b"):
                    nw = c.height() * asp;  mx = c.center().x()
                    c.setLeft(mx - nw / 2); c.setRight(mx + nw / 2)
                elif mode in ("l", "r"):
                    nh = c.width() / asp;   my = c.center().y()
                    c.setTop(my - nh / 2);  c.setBottom(my + nh / 2)
                elif mode in ("tl", "bl"):
                    c.setLeft(c.right()  - c.height() * asp)
                elif mode in ("tr", "br"):
                    c.setRight(c.left()  + c.height() * asp)

        if c.width()  < 20: c.setWidth(20)
        if c.height() < 20: c.setHeight(20)

        self._crop_w = c
        self._clamp_crop_w()
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_mode = ""
            pos    = QPointF(e.pos())
            handle = self._hit_handle(pos)
            if handle:
                self.setCursor(self._cursor_for(handle))
            elif self._crop_w.contains(pos):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, e):
        # Zoom image around cursor; crop frame stays fixed in widget coords
        factor    = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        new_scale = max(0.5, min(10.0, self._view_scale * factor))
        mouse_img        = self._widget_to_img(QPointF(e.pos()))
        self._view_scale = new_scale
        ds               = self._display_scale()
        mx, my           = float(e.pos().x()), float(e.pos().y())
        self._center_img = QPointF(
            mouse_img.x() - (mx - self.width()  / 2) / ds,
            mouse_img.y() - (my - self.height() / 2) / ds,
        )
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._crop_init:
            self._init_crop()
        else:
            self._clamp_crop_w()
        self.update()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _clamp_crop_w(self):
        c  = self._crop_w
        ww = float(self.width())
        wh = float(self.height())
        if c.right()  > ww: c.translate(ww - c.right(),  0)
        if c.left()   < 0:  c.translate(-c.left(),        0)
        if c.bottom() > wh: c.translate(0, wh - c.bottom())
        if c.top()    < 0:  c.translate(0, -c.top())
        c.setLeft(max(0.0, c.left()));  c.setRight(min(ww, c.right()))
        c.setTop(max(0.0, c.top()));    c.setBottom(min(wh, c.bottom()))
        self._crop_w = c

    def get_crop_rect(self) -> QRect:
        """Crop result in original image pixel coordinates."""
        tl = self._widget_to_img(self._crop_w.topLeft())
        br = self._widget_to_img(self._crop_w.bottomRight())
        x  = max(0, int(tl.x()))
        y  = max(0, int(tl.y()))
        x2 = min(self._orig_w, int(br.x()))
        y2 = min(self._orig_h, int(br.y()))
        return QRect(x, y, max(1, x2 - x), max(1, y2 - y))

    def reset_crop(self):
        """Reset frame to full image at fit-to-window zoom."""
        self._crop_init = False
        self._init_crop()
        self.update()


class CropDialog(QDialog):
    """Crop dialog with aspect ratio presets, drag handles, pan, and zoom."""

    def __init__(self, image_path: str, parent=None, img_bgr: '_np.ndarray | None' = None,
                 title: str = "Ritaglia foto",
                 hint_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setStyleSheet(BASE_STYLE)

        if img_bgr is not None:
            img = img_bgr
        else:
            img = _cv2.imread(image_path)
            if img is None:
                img = _np.zeros((100, 100, 3), dtype=_np.uint8)
        oh, ow = img.shape[:2]
        self._orig_w, self._orig_h = ow, oh

        rgb    = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        qimg   = QImage(rgb.data, ow, oh, 3 * ow, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Instruction bar ───────────────────────────────────────────────────
        hint = QLabel(
            "Bordi/angoli: ridimensiona  ·  Dentro: sposta il riquadro  ·  "
            "Fuori: sposta l'immagine  ·  Rotellina: zoom"
        )
        hint.setStyleSheet("color: #777; font-size: 11px;")
        lay.addWidget(hint)

        # ── Aspect ratio chips (scrollable row) ───────────────────────────────
        scroll = QScrollArea()
        scroll.setFixedHeight(44)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:horizontal { height: 4px; background: #222; }"
            "QScrollBar::handle:horizontal { background: #555; border-radius: 2px; }"
        )
        chip_widget = QWidget()
        chip_widget.setStyleSheet("background: transparent;")
        chip_lay = QHBoxLayout(chip_widget)
        chip_lay.setContentsMargins(0, 4, 0, 4)
        chip_lay.setSpacing(5)

        self._aspect_btns: list[QPushButton] = []
        for label, ratio in _ASPECT_PRESETS:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(_PRESET_BTN_STYLE)
            btn.clicked.connect(lambda _c, r=ratio, b=btn: self._set_aspect(r, b))
            self._aspect_btns.append(btn)
            chip_lay.addWidget(btn)
        chip_lay.addStretch()
        scroll.setWidget(chip_widget)
        lay.addWidget(scroll)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._canvas = _CropCanvas(pixmap, ow, oh, self)
        lay.addWidget(self._canvas, 1)

        # ── Dialog buttons ────────────────────────────────────────────────────
        btns      = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btn_reset = btns.addButton("Usa foto intera", QDialogButtonBox.ResetRole)
        btn_reset.clicked.connect(self._canvas.reset_crop)
        btn_reset.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(btns)

        screen = QApplication.primaryScreen()
        avail  = screen.availableGeometry() if screen else None
        max_w  = int(avail.width()  * 0.88) if avail else 1100
        max_h  = int(avail.height() * 0.88) if avail else 850
        self.resize(max_w, max_h)

        # ── Restore last used aspect ratio ────────────────────────────────────
        last_label = _load_settings().get("last_crop_aspect_label", "Libero")
        for i, (lbl, ratio) in enumerate(_ASPECT_PRESETS):
            is_match = lbl == last_label
            self._aspect_btns[i].setChecked(is_match)
            if is_match and ratio != None:  # None = Libero, no reshape needed
                actual: float | None = (
                    float(ow) / max(oh, 1) if ratio == -1.0 else ratio
                )
                self._canvas.set_aspect(actual)

    # ── Aspect ratio ──────────────────────────────────────────────────────────

    def _set_aspect(self, ratio: float | None, sender: QPushButton):
        for btn in self._aspect_btns:
            btn.setChecked(btn is sender)
        actual: float | None = (
            float(self._orig_w) / max(self._orig_h, 1) if ratio == -1.0 else ratio
        )
        self._canvas.set_aspect(actual)
        label = next((lbl for lbl, r in _ASPECT_PRESETS if r == ratio), "Libero")
        _save_settings({"last_crop_aspect_label": label})

    # ── Result ────────────────────────────────────────────────────────────────

    def crop_rect(self) -> 'QRect | None':
        r = self._canvas.get_crop_rect()
        if r.x() == 0 and r.y() == 0 and r.width() == self._orig_w and r.height() == self._orig_h:
            return None
        return r

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.accept()
        else:
            super().keyPressEvent(event)


# ── Watermark setup dialog ────────────────────────────────────────────────────

class WatermarkSetupDialog(QDialog):
    def __init__(self, parent=None, existing: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Impostazioni firma")
        self.setModal(True)
        self.setStyleSheet(BASE_STYLE)
        self.setMinimumWidth(500)

        existing = existing or {}
        self._paths = {
            "dark":  str(existing.get("dark") or ""),
            "light": str(existing.get("light") or ""),
        }

        lay = QVBoxLayout(self)
        lay.setSpacing(14)
        lay.setContentsMargins(18, 18, 18, 18)

        info = QLabel(
            "Puoi caricare una sola firma (usata sempre) oppure una versione scura "
            "e una chiara: il programma sceglierà automaticamente quella con più "
            "contrasto rispetto allo sfondo della foto."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #bbb; font-size: 12px; margin-bottom: 4px;")
        lay.addWidget(info)

        for key, label in (("dark", "Firma scura / singola"), ("light", "Firma chiara  (opzionale)")):
            row = QHBoxLayout()
            row.setSpacing(8)
            btn_pick = QPushButton(f"📁  {label}")
            btn_pick.setFixedHeight(32)
            btn_pick.clicked.connect(lambda _, k=key: self._pick(k))
            row.addWidget(btn_pick)

            lbl = QLabel(self._paths[key] if self._paths[key] else "Nessuna")
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            lbl.setWordWrap(True)
            setattr(self, f"_lbl_{key}", lbl)
            row.addWidget(lbl, 1)

            btn_rm = QPushButton("✕")
            btn_rm.setFixedSize(28, 32)
            btn_rm.setToolTip("Rimuovi")
            btn_rm.setEnabled(bool(self._paths[key]))
            btn_rm.clicked.connect(lambda _, k=key: self._remove(k))
            setattr(self, f"_btn_rm_{key}", btn_rm)
            row.addWidget(btn_rm)

            lay.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _pick(self, key: str):
        path, _ = QFileDialog.getOpenFileName(self, "Seleziona firma PNG", "", "PNG (*.png)")
        if not path:
            return
        self._paths[key] = path
        getattr(self, f"_lbl_{key}").setText(path)
        getattr(self, f"_btn_rm_{key}").setEnabled(True)

    def _remove(self, key: str):
        self._paths[key] = ""
        label = "Nessuna"
        getattr(self, f"_lbl_{key}").setText(label)
        getattr(self, f"_btn_rm_{key}").setEnabled(False)

    def _on_accept(self):
        if not self._paths["dark"] and not self._paths["light"]:
            QMessageBox.warning(self, "Firma richiesta", "Seleziona almeno una firma PNG.")
            return
        self.accept()

    def paths(self) -> dict:
        return self._paths


# ── Logo position dialog ──────────────────────────────────────────────────────

class LogoPositionDialog(QDialog):
    def __init__(self, parent=None, pos_h: str = "bot-right", pos_v: str = "bot-right",
                 size_h: int = 15, size_v: int = 15):
        super().__init__(parent)
        self.setWindowTitle("Posizione logo")
        self.setModal(True)
        self.setStyleSheet(BASE_STYLE)

        self._pos_h = pos_h
        self._pos_v = pos_v
        self._btns: dict[str, dict] = {"h": {}, "v": {}}

        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        lay.setContentsMargins(18, 18, 18, 18)

        grids_row = QHBoxLayout()
        grids_row.setSpacing(24)

        for axis, label, init_pos, init_size in (
            ("h", "Foto orizzontali", pos_h, size_h),
            ("v", "Foto verticali",   pos_v, size_v),
        ):
            col = QVBoxLayout()
            col.setSpacing(8)

            hdr = QLabel(label)
            hdr.setStyleSheet("color:#ddd; font-weight:bold; font-size:13px;")
            col.addWidget(hdr)

            grid = QGridLayout()
            grid.setSpacing(4)
            for idx, (key, icon) in enumerate(_LOGO_POS_GRID):
                btn = QPushButton(icon)
                btn.setFixedSize(48, 48)
                btn.setCheckable(True)
                btn.setChecked(key == init_pos)
                btn.setStyleSheet(_LOGO_POS_STYLE)
                btn.setFocusPolicy(Qt.NoFocus)
                btn.clicked.connect(lambda _, k=key, ax=axis: self._pick(ax, k))
                self._btns[axis][key] = btn
                grid.addWidget(btn, idx // 3, idx % 3)
            col.addLayout(grid)

            size_row = QHBoxLayout()
            size_row.setSpacing(6)
            size_row.addWidget(QLabel("Dimensione:"))
            spin = QSpinBox()
            spin.setRange(3, 60)
            spin.setValue(init_size)
            spin.setSuffix(" %")
            spin.setFixedWidth(72)
            setattr(self, f"_spin_{axis}", spin)
            size_row.addWidget(spin)
            size_row.addStretch()
            col.addLayout(size_row)

            grids_row.addLayout(col)

        lay.addLayout(grids_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _pick(self, axis: str, key: str):
        for k, btn in self._btns[axis].items():
            btn.setChecked(k == key)
        if axis == "h":
            self._pos_h = key
        else:
            self._pos_v = key

    def result_pos_h(self) -> str:   return self._pos_h
    def result_pos_v(self) -> str:   return self._pos_v
    def result_size_h(self) -> int:  return self._spin_h.value()
    def result_size_v(self) -> int:  return self._spin_v.value()


# ── Image view ────────────────────────────────────────────────────────────────

class ImageView(QGraphicsView):
    wm_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setStyleSheet("background: #1a1a1a; border: none;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.NoFocus)
        self._item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0

        self._tint_strength = 0.0
        self._snake_active  = False
        self._snake_offset  = 0.0
        self._snake_timer   = QTimer(self)
        self._snake_timer.timeout.connect(self._snake_tick)
        self._wm_preview: QPixmap | None = None
        self._wm_size_frac_h: float = 0.22   # landscape
        self._wm_size_frac_v: float = 0.22   # portrait
        self._logo_preview: QPixmap | None = None
        self._logo_pos:      str   = "bot-right"
        self._logo_size_frac: float = 0.15

    # ── Watermark preview ─────────────────────────────────────────────────────

    def set_watermark_preview(self, pixmap: 'QPixmap | None'):
        self._wm_preview = pixmap
        self.viewport().update()

    def set_watermark_sizes(self, frac_h: float, frac_v: float):
        self._wm_size_frac_h = frac_h
        self._wm_size_frac_v = frac_v
        self.viewport().update()

    def set_logo_preview(self, pixmap: 'QPixmap | None', pos: str, size_frac: float):
        self._logo_preview  = pixmap
        self._logo_pos      = pos
        self._logo_size_frac = size_frac
        self.viewport().update()

    def current_pixmap(self) -> 'QPixmap | None':
        return self._item.pixmap() if self._item else None

    # ── Tint property (animated by QPropertyAnimation) ────────────────────────

    @pyqtProperty(float)
    def tintStrength(self) -> float:
        return self._tint_strength

    @tintStrength.setter
    def tintStrength(self, v: float):
        self._tint_strength = v
        self.viewport().update()

    # ── Snake border animation ─────────────────────────────────────────────────

    def start_snake(self):
        if not self._snake_active:
            self._snake_offset = 0.0
            self._snake_active = True
        if not self._snake_timer.isActive():
            self._snake_timer.start(16)

    def stop_snake(self):
        if self._snake_active:
            self._snake_active = False
            self._snake_timer.stop()
            self.viewport().update()

    def _snake_tick(self):
        self._snake_offset += 0.75
        self.viewport().update()

    # ── Overlay drawing (tint + snake + watermark) ───────────────────────────

    def drawForeground(self, painter, rect):
        has_tint  = self._tint_strength > 0.0
        has_snake = self._snake_active
        has_wm    = self._wm_preview    is not None and self._item is not None
        has_logo  = self._logo_preview  is not None and self._item is not None
        if not has_tint and not has_snake and not has_wm and not has_logo:
            return

        painter.save()
        painter.resetTransform()
        vp = self.viewport()
        w, h = vp.width(), vp.height()

        if has_tint:
            painter.fillRect(0, 0, w, h,
                             QColor(220, 20, 20, int(self._tint_strength * 170)))

        if has_snake and self._item is not None:
            PW     = 4.0
            HEAD_R = 8.0
            PAD    = 6      # padding around the actual photo edge

            # Map the photo's scene bounding rect into viewport pixel coordinates
            vp_rect = self.mapFromScene(
                self._item.sceneBoundingRect()
            ).boundingRect()

            x  = vp_rect.x() - PAD
            y  = vp_rect.y() - PAD
            rw = vp_rect.width()  + 2 * PAD
            rh = vp_rect.height() + 2 * PAD

            perim_px = 2 * (rw + rh)
            pu       = perim_px / PW

            snake_pu     = pu * 0.16
            gap_pu       = pu * 0.5 - snake_pu
            snake_len_px = snake_pu * PW

            # Continuous path so the dash flows smoothly around all corners
            path = QPainterPath()
            path.moveTo(x,        y)
            path.lineTo(x + rw,   y)
            path.lineTo(x + rw,   y + rh)
            path.lineTo(x,        y + rh)
            path.lineTo(x,        y)

            pen = QPen(QColor(50, 220, 100), PW)
            pen.setStyle(Qt.CustomDashLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setDashPattern([snake_pu, gap_pu])
            pen.setDashOffset(-(self._snake_offset % (pu * 0.5)))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

            # Head circle at the leading edge of each snake
            tail_px = (self._snake_offset * PW) % perim_px

            def pt_on_rect(dist_px):
                d = dist_px % perim_px
                if d <= rw:  return QPointF(x + d,       y)
                d -= rw
                if d <= rh:  return QPointF(x + rw,      y + d)
                d -= rh
                if d <= rw:  return QPointF(x + rw - d,  y + rh)
                d -= rw
                return            QPointF(x,             y + rh - d)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(80, 255, 130))
            for phase in (0.0, perim_px * 0.5):
                head_px = (tail_px + snake_len_px + phase) % perim_px
                painter.drawEllipse(pt_on_rect(head_px), HEAD_R, HEAD_R)

        if has_wm:
            vp_rect = self.mapFromScene(
                self._item.sceneBoundingRect()
            ).boundingRect()
            iw = vp_rect.width()
            ih = vp_rect.height()
            px   = self._item.pixmap()
            frac = self._wm_size_frac_h if px.width() >= px.height() else self._wm_size_frac_v
            wm_w = max(1, int(iw * frac))
            wm_h = max(1, int(self._wm_preview.height() * wm_w
                               / max(1, self._wm_preview.width())))
            margin = max(1, int(ih * 0.03))
            wx = int(vp_rect.x() + (iw - wm_w) / 2)
            wy = int(vp_rect.y() + ih - wm_h - margin)
            scaled_wm = self._wm_preview.scaled(
                wm_w, wm_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(wx, wy, scaled_wm)

        if has_logo:
            vp_rect = self.mapFromScene(
                self._item.sceneBoundingRect()
            ).boundingRect()
            iw = int(vp_rect.width())
            ih = int(vp_rect.height())
            lw = max(1, int(iw * self._logo_size_frac))
            lh = max(1, int(self._logo_preview.height() * lw
                             / max(1, self._logo_preview.width())))
            margin = max(1, int(min(iw, ih) * 0.02))
            ox, oy = _logo_xy(self._logo_pos, iw, ih, lw, lh, margin)
            lx = int(vp_rect.x()) + ox
            ly = int(vp_rect.y()) + oy
            scaled_logo = self._logo_preview.scaled(
                lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(lx, ly, scaled_logo)

        painter.restore()

    def load(self, pixmap: QPixmap):
        self._scene.clear()
        self._item = QGraphicsPixmapItem(pixmap)
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self._fit()

    def show_placeholder(self, text: str = ""):
        self._scene.clear()
        self._item = None
        self._scene.setSceneRect(QRectF(0, 0, 1, 1))
        self.resetTransform()

    def _fit(self):
        if self._item:
            self.fitInView(self._item.boundingRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def zoom_in(self):    self._scale(1.25)
    def zoom_out(self):   self._scale(0.80)
    def zoom_reset(self): self._fit()

    def _scale(self, factor: float):
        new = max(0.05, min(10.0, self._zoom * factor))
        self.scale(new / self._zoom, new / self._zoom)
        self._zoom = new

    def wheelEvent(self, event):
        if self._item:
            self._scale(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def _wm_viewport_rect(self):
        """Returns the watermark bounding rect in viewport coords, or None."""
        if self._wm_preview is None or self._item is None:
            return None
        vp_rect = self.mapFromScene(self._item.sceneBoundingRect()).boundingRect()
        iw = vp_rect.width()
        ih = vp_rect.height()
        px   = self._item.pixmap()
        frac = self._wm_size_frac_h if px.width() >= px.height() else self._wm_size_frac_v
        wm_w = max(1, int(iw * frac))
        wm_h = max(1, int(self._wm_preview.height() * wm_w
                           / max(1, self._wm_preview.width())))
        margin = max(1, int(ih * 0.03))
        wx = int(vp_rect.x() + (iw - wm_w) / 2)
        wy = int(vp_rect.y() + ih - wm_h - margin)
        return QRect(wx, wy, wm_w, wm_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._wm_preview is not None:
            rect = self._wm_viewport_rect()
            if rect and rect.contains(event.pos()):
                self.wm_clicked.emit()
                return
        if event.button() == Qt.MiddleButton:
            self._fit()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        NAV = (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
               Qt.Key_A, Qt.Key_D, Qt.Key_Space, Qt.Key_I,
               Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus, Qt.Key_0,
               Qt.Key_BracketLeft, Qt.Key_BracketRight)
        if event.key() in NAV:
            event.ignore()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._item:
            self._fit()

# ── Info panel ────────────────────────────────────────────────────────────────

class InfoPanel(QWidget):
    """Toggleable right-side panel showing EXIF metadata and GPS location."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(272)
        self.setStyleSheet(
            "QWidget { background: #1e1e2a; }"
            "QScrollArea { background: #1e1e2a; border: none; }"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._inner = QWidget()
        self._vbox  = QVBoxLayout(self._inner)
        self._vbox.setContentsMargins(16, 12, 16, 12)
        self._vbox.setSpacing(1)
        scroll.setWidget(self._inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        self._vals: dict[str, QLabel] = {}
        self._gps_group: list[QLabel] = []   # widgets to hide when no GPS
        self._build_layout()
        self.clear()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        def section(icon: str, title: str) -> QLabel:
            lbl = QLabel(f"  {icon}  {title.upper()}")
            lbl.setStyleSheet(
                "color: #666; font-size: 10px; font-weight: bold;"
                "padding-top: 14px; padding-bottom: 4px;"
                "border-bottom: 1px solid #2e2e3e;"
            )
            self._vbox.addWidget(lbl)
            return lbl

        def row(key: str, label: str) -> QLabel:
            k = QLabel(label)
            k.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
            v = QLabel("—")
            v.setStyleSheet("color: #e8e8e8; font-size: 12px;")
            v.setWordWrap(True)
            self._vbox.addWidget(k)
            self._vbox.addWidget(v)
            self._vals[key] = v
            return k

        section("📁", "File")
        row("filename",    "Nome")
        row("filesize",    "Dimensione")
        row("dimensions",  "Risoluzione")
        row("date_taken",  "Data scatto")

        section("📷", "Fotocamera")
        row("camera",        "Modello")
        row("lens",          "Obiettivo")
        row("focal_length",  "Lunghezza focale")
        row("aperture",      "Diaframma")
        row("shutter_speed", "Esposizione")
        row("iso",           "ISO")

        gps_header      = section("📍", "Posizione GPS")
        gps_coords_key  = row("gps_coords",   "Coordinate")
        gps_loc_key     = row("gps_location", "Luogo")
        self._gps_group = [
            gps_header, gps_coords_key, self._vals["gps_coords"],
            gps_loc_key, self._vals["gps_location"],
        ]

        self._vbox.addStretch()

    # ── public API ────────────────────────────────────────────────────────────

    def clear(self):
        for lbl in self._vals.values():
            lbl.setText("—")

    def update_info(self, info: dict):
        self.clear()
        self._set("filename", info.get("filename", ""))

        size = info.get("filesize", 0)
        self._set("filesize",
                  f"{size/1e6:.1f} MB" if size >= 1e6 else
                  f"{size/1e3:.0f} KB" if size >= 1e3 else f"{size} B")

        w, h = info.get("width", 0), info.get("height", 0)
        if w and h:
            self._set("dimensions", f"{w} × {h}  ({w*h/1e6:.1f} MP)")

        dt: datetime.datetime | None = info.get("date_taken")
        if dt:
            self._set("date_taken", dt.strftime("%d %b %Y  %H:%M"))

        make  = info.get("camera_make", "")
        model = info.get("camera_model", "")
        if make and model.startswith(make):
            self._set("camera", model)
        else:
            self._set("camera", f"{make} {model}".strip())

        self._set("lens",          info.get("lens", ""))
        self._set("focal_length",  info.get("focal_length", ""))
        self._set("aperture",      info.get("aperture", ""))
        self._set("shutter_speed", info.get("shutter_speed", ""))
        self._set("iso",           info.get("iso", ""))

        lat = info.get("gps_lat")
        lon = info.get("gps_lon")
        has_gps = lat is not None and lon is not None
        for w in self._gps_group:
            w.setVisible(has_gps)
        if has_gps:
            self._set("gps_coords",   f"{lat:.5f},  {lon:.5f}")
            self._set("gps_location", "Ricerca in corso...")

    def set_location(self, location: str):
        self._set("gps_location", location or "—")

    def _set(self, key: str, text: str):
        if key in self._vals:
            self._vals[key].setText(text or "—")

# ── Separator helper ──────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFixedHeight(28)
    line.setStyleSheet("background: #444; margin: 4px 2px;")
    line.setFixedWidth(1)
    return line

# ── Filemail dialogs ──────────────────────────────────────────────────────────

class FilemailSettingsDialog(QDialog):
    """First-run (or on-demand) dialog to capture email and optional API key."""

    def __init__(self, parent, email: str = "", api_key: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Impostazioni Filemail")
        self.setMinimumWidth(440)
        self.setStyleSheet(BASE_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        info = QLabel(
            "Per condividere le foto via Filemail inserisci la tua email "
            "(viene usata come mittente del link).\n\n"
            "La chiave API è opzionale — richiedila gratis su filemail.com "
            "nella sezione Account → API."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(info)

        layout.addWidget(QLabel("Email mittente *"))
        self._email = QLineEdit(email)
        self._email.setPlaceholderText("tua@email.com")
        layout.addWidget(self._email)

        layout.addWidget(QLabel("Chiave API (opzionale)"))
        self._key = QLineEdit(api_key)
        self._key.setPlaceholderText("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        layout.addWidget(self._key)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        if not self._email.text().strip():
            QMessageBox.warning(self, "Email richiesta",
                                "Inserisci la tua email per continuare.")
            return
        self.accept()

    def email(self) -> str:
        return self._email.text().strip()

    def api_key(self) -> str:
        return self._key.text().strip()


class ShareLinkDialog(QDialog):
    """Shows the Filemail download link with copy and open-in-browser buttons."""

    def __init__(self, parent, url: str):
        super().__init__(parent)
        self.setWindowTitle("Foto condivise!")
        self.setMinimumWidth(520)
        self.setStyleSheet(BASE_STYLE)
        self._url = url

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("✅  Le foto sono state caricate su Filemail!")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00c870;")
        layout.addWidget(title)

        sub = QLabel("Condividi questo link con i tuoi amici (disponibile per 7 giorni):")
        sub.setStyleSheet("color: #aaa;")
        layout.addWidget(sub)

        link_row = QHBoxLayout()
        self._link_edit = QLineEdit(url)
        self._link_edit.setReadOnly(True)
        self._link_edit.setStyleSheet(
            "background: #1a1a1a; color: #f0f0f0; padding: 6px;"
            "border: 1px solid #444; border-radius: 4px;"
        )
        link_row.addWidget(self._link_edit, 1)

        self._btn_copy = QPushButton("📋  Copia")
        self._btn_copy.setFixedHeight(34)
        self._btn_copy.clicked.connect(self._copy_link)
        link_row.addWidget(self._btn_copy)
        layout.addLayout(link_row)

        btn_row = QHBoxLayout()
        btn_browser = QPushButton("🌐  Apri nel browser")
        btn_browser.clicked.connect(self._open_browser)
        btn_row.addWidget(btn_browser)
        btn_row.addStretch()

        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet(ACCENT_STYLE)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _copy_link(self):
        QApplication.clipboard().setText(self._url)
        self._btn_copy.setText("✅  Copiato!")
        self._btn_copy.setStyleSheet(
            "QPushButton { background: #005a9e; color: #fff; border-radius: 4px; }"
        )

    def _open_browser(self):
        QDesktopServices.openUrl(QUrl(self._url))


# ── Main window ───────────────────────────────────────────────────────────────

class PhotoSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.photos: list[Path]  = []
        self.current_index       = -1
        self.selected: set[int]  = set()
        self._folder: Path | None = None

        self._thumb_loader:   ThumbnailLoader | None  = None
        self._img_loader:     ImageLoader | None       = None
        self._exif_loader:    ExifLoader | None        = None
        self._geo_loader:     GeocoderThread | None    = None

        # Debounce rapid arrow-key navigation: update the index immediately
        # but only start the (expensive) image-load thread after the user
        # pauses for 150 ms. Without this, holding an arrow key spawns a new
        # ImageLoader thread for every keypress — each decoding a full-res
        # JPEG — causing rapid memory exhaustion and an OOM crash.
        self._nav_timer = QTimer(self)
        self._nav_timer.setSingleShot(True)
        self._nav_timer.setInterval(150)
        self._nav_timer.timeout.connect(self._load_current_image)
        self._rotate_worker:  RotateWorker | None      = None
        self._rotate_thumb_idx: int | None             = None  # needs thumb regen
        self._share_worker:   FilemailUploader | None  = None
        self._share_progress: QProgressDialog | None   = None
        self._share_links:    list[str]                = []   # accumulated download URLs
        self._share_batch2:   list                     = []   # second-batch for split transfers
        self._share_email:    str                      = ""
        self._share_apikey:   str                      = ""
        self._share_n_batches: int                     = 1
        self._dest_folder:    str                       = DEST_FOLDER
        self._discard_history: list[tuple[Path, Path, int]] = []
        self._space_press_idx: int = -1
        self._current_location: str = ""
        self._watermark_enabled: bool   = False
        # global color preference: None=auto, "dark", "light"
        self._wm_override: str | None = None
        # per-photo click-override: {idx: "dark" | "light"}; absent = use auto
        self._wm_photo_override: dict[int, str] = {}
        self.logo_marked:   set[int]    = set()
        self._logo_path:    str | None  = None
        self._logo_bgra                 = None   # numpy array, loaded once per session
        self._logo_pos_h:   str         = "bot-right"
        self._logo_pos_v:   str         = "bot-right"
        self._logo_size_h:  int         = 15
        self._logo_size_v:  int         = 15

        self._space_timer = QTimer(self)
        self._space_timer.setSingleShot(True)
        self._space_timer.setInterval(2000)
        self._space_timer.timeout.connect(self._discard_current)

        self._build_ui()
        self._connect()

        self._tint_anim = QPropertyAnimation(self.image_view, b"tintStrength")
        self._tint_anim.setDuration(2000)
        self._tint_anim.setStartValue(0.0)
        self._tint_anim.setEndValue(1.0)

        self._sync_ui()
        self._restore_last_session()

    # ── Session restore ───────────────────────────────────────────────────────

    def _restore_last_session(self):
        settings = _load_settings()

        # Restore info panel open/closed state before loading any folder.
        if settings.get("info_panel_open", False):
            self.btn_info.setChecked(True)
            self._toggle_info()

        # Restore watermark sizes, toggle state and color override.
        self.image_view.set_watermark_sizes(
            settings.get("watermark_size_pct_h", 22) / 100.0,
            settings.get("watermark_size_pct_v", 22) / 100.0,
        )
        saved_ov = settings.get("wm_override", "off")
        self._wm_override = None if saved_ov == "off" else saved_ov
        if settings.get("watermark_enabled", False):
            wms = _load_watermarks()
            if wms["dark"] or wms["light"]:
                self._watermark_enabled = True
                self._apply_watermark_btn_style()


    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("PhotoSelector")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 820)
        self.setStyleSheet(BASE_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        root_vbox = QVBoxLayout(root)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────────────────────
        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet("background: #272727;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(6)

        self.btn_open = QPushButton("📂  Apri cartella")
        self.btn_open.setFixedHeight(36)
        self.btn_open.setShortcut(QKeySequence("Ctrl+O"))

        self.btn_prev = QPushButton("‹")
        self.btn_prev.setFixedSize(36, 36)

        self.lbl_counter = QLabel("—")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        self.lbl_counter.setMinimumWidth(90)
        self.lbl_counter.setStyleSheet("color: #aaa;")

        self.btn_next = QPushButton("›")
        self.btn_next.setFixedSize(36, 36)

        self.lbl_name = QLabel("")
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet("color: #ddd;")
        self.lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.btn_select = QPushButton("☆  Seleziona  [Space]")
        self.btn_select.setFixedHeight(36)
        self.btn_select.setMinimumWidth(170)

        self.btn_copy = QPushButton("📋  Copia selezionate  (0)")
        self.btn_copy.setFixedHeight(36)
        self.btn_copy.setStyleSheet(ACCENT_STYLE)
        self.btn_copy.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_copy.customContextMenuRequested.connect(self._set_dest_folder)

        self.btn_share = QPushButton("📤  Condividi")
        self.btn_share.setFixedHeight(36)
        self.btn_share.setStyleSheet(SHARE_STYLE)
        self.btn_share.setToolTip(
            "Carica la cartella 'selezionate' su Filemail e mostra il link da condividere\n"
            "Tasto destro → Impostazioni email"
        )
        self.btn_share.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_share.customContextMenuRequested.connect(self._share_settings)

        self.btn_info = QPushButton("ℹ  Info")
        self.btn_info.setFixedHeight(36)
        self.btn_info.setCheckable(True)

        self.btn_rot_left  = QPushButton("Rot. SX")
        self.btn_rot_left.setFixedHeight(36)
        self.btn_rot_left.setMinimumWidth(72)
        self.btn_rot_left.setToolTip("Ruota a sinistra  [[]")

        self.btn_rot_right = QPushButton("Rot. DX")
        self.btn_rot_right.setFixedHeight(36)
        self.btn_rot_right.setMinimumWidth(72)
        self.btn_rot_right.setToolTip("Ruota a destra  []]")

        self.btn_crop = QPushButton("✂  Ritaglia")
        self.btn_crop.setFixedHeight(36)
        self.btn_crop.setToolTip(
            "Ritaglia la foto corrente: il risultato viene salvato come\n"
            "nuovo file accanto all'originale, senza modificarlo"
        )

        self.btn_zout  = QPushButton("−")
        self.btn_zout.setFixedSize(36, 36)
        self.btn_zreset = QPushButton("⊡")
        self.btn_zreset.setFixedSize(36, 36)
        self.btn_zreset.setToolTip("Adatta alla finestra  [0]")
        self.btn_zin   = QPushButton("+")
        self.btn_zin.setFixedSize(36, 36)

        self.btn_watermark = QPushButton("✍  Firma")
        self.btn_watermark.setFixedHeight(36)
        self.btn_watermark.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_watermark.customContextMenuRequested.connect(self._watermark_context_menu)
        self.btn_watermark.setToolTip(
            "Applica la firma alle foto durante la copia\n"
            "Tasto destro → Modifica / Rimuovi firma"
        )

        self.btn_logo = QPushButton("🖼  Logo")
        self.btn_logo.setFixedHeight(36)
        self.btn_logo.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_logo.customContextMenuRequested.connect(self._logo_context_menu)
        self.btn_logo.setToolTip(
            "Applica un logo alle foto durante la copia (sessione corrente)\n"
            "Tasto destro → Cambia logo / Cambia posizione"
        )

        for w in [self.btn_open,
                  _vsep(), self.btn_prev, self.lbl_counter, self.btn_next,
                  _vsep(), self.lbl_name,
                  _vsep(), self.btn_select, self.btn_copy, self.btn_share,
                  _vsep(), self.btn_info,
                  _vsep(), self.btn_rot_left, self.btn_rot_right, self.btn_crop,
                  _vsep(), self.btn_watermark, self.btn_logo,
                  _vsep(), self.btn_zout, self.btn_zreset, self.btn_zin]:
            if isinstance(w, QPushButton):
                w.setFocusPolicy(Qt.NoFocus)
            row.addWidget(w)

        # ── Thumbnail loading progress bar (hidden until loading starts) ──────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        # ── Content: image area + info panel side by side ─────────────────────
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_row = QHBoxLayout(content)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        # image area
        img_area = QWidget()
        img_area.setStyleSheet("background: #1a1a1a;")
        img_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_row = QHBoxLayout(img_area)
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.setSpacing(0)

        self.btn_img_prev = QPushButton("‹")
        self.btn_img_prev.setFixedSize(46, 100)
        self.btn_img_prev.setStyleSheet(NAV_ARROW_STYLE)
        self.btn_img_prev.setFocusPolicy(Qt.NoFocus)

        self.image_view = ImageView()
        self.image_view.show_placeholder("Apri una cartella per iniziare  —  Ctrl+O")

        self.btn_img_next = QPushButton("›")
        self.btn_img_next.setFixedSize(46, 100)
        self.btn_img_next.setStyleSheet(NAV_ARROW_STYLE)
        self.btn_img_next.setFocusPolicy(Qt.NoFocus)

        img_row.addWidget(self.btn_img_prev)
        img_row.addWidget(self.image_view, 1)
        img_row.addWidget(self.btn_img_next)

        # info panel (hidden by default)
        self.info_panel = InfoPanel()
        self.info_panel.hide()

        # divider line between image and info panel
        self._info_divider = QFrame()
        self._info_divider.setFrameShape(QFrame.VLine)
        self._info_divider.setStyleSheet("background: #2e2e3e;")
        self._info_divider.setFixedWidth(1)
        self._info_divider.hide()

        content_row.addWidget(img_area, 1)
        content_row.addWidget(self._info_divider)
        content_row.addWidget(self.info_panel)

        # ── Film strip ────────────────────────────────────────────────────────
        self.filmstrip = FilmStrip()

        # ── Status bar ────────────────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Pronto — apri una cartella per iniziare")

        root_vbox.addWidget(bar)
        root_vbox.addWidget(self.progress_bar)
        root_vbox.addWidget(content, 1)
        root_vbox.addWidget(self.filmstrip)

    # ── Connections ───────────────────────────────────────────────────────────

    def _connect(self):
        self.btn_open.clicked.connect(self._open_folder)
        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)
        self.btn_img_prev.clicked.connect(self._prev)
        self.btn_img_next.clicked.connect(self._next)
        self.btn_select.clicked.connect(self._toggle_select)
        self.btn_copy.clicked.connect(self._copy_selected)
        self.btn_share.clicked.connect(self._share)
        self.btn_info.clicked.connect(self._toggle_info)
        self.btn_rot_left.clicked.connect(lambda: self._rotate(clockwise=False))
        self.btn_rot_right.clicked.connect(lambda: self._rotate(clockwise=True))
        self.btn_crop.clicked.connect(self._crop_current)
        self.btn_watermark.clicked.connect(self._toggle_watermark)
        self.image_view.wm_clicked.connect(self._cycle_wm_photo_color)
        self.btn_logo.clicked.connect(self._toggle_logo)
        self.btn_zin.clicked.connect(self.image_view.zoom_in)
        self.btn_zout.clicked.connect(self.image_view.zoom_out)
        self.btn_zreset.clicked.connect(self.image_view.zoom_reset)
        self.filmstrip.navigate.connect(self._go_to)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        _mark_clean_exit()
        super().closeEvent(event)

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space:
            if not event.isAutoRepeat() and self.current_index >= 0:
                self._space_press_idx = self.current_index
                self._space_timer.start()
                self._tint_anim.start()
                self.status.showMessage(
                    "  Tieni premuto Spazio per 2 secondi per spostare in Scartate..."
                )
        elif k in (Qt.Key_Right, Qt.Key_D):
            self._next()
        elif k in (Qt.Key_Left, Qt.Key_A):
            self._prev()
        elif k == Qt.Key_I:
            self.btn_info.toggle()
            self._toggle_info()
        elif k == Qt.Key_BracketLeft:
            self._rotate(clockwise=False)
        elif k == Qt.Key_BracketRight:
            self._rotate(clockwise=True)
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.image_view.zoom_in()
        elif k == Qt.Key_Minus:
            self.image_view.zoom_out()
        elif k == Qt.Key_0:
            self.image_view.zoom_reset()
        elif k == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self._undo_discard()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self._space_timer.isActive():
                self._space_timer.stop()
                self._tint_anim.stop()
                self.image_view.tintStrength = 0.0
                self._toggle_select()
            # timer already fired → discard already happened, do nothing
        else:
            super().keyReleaseEvent(event)

    # ── Info panel toggle ─────────────────────────────────────────────────────

    def _toggle_info(self):
        visible = self.btn_info.isChecked()
        self.info_panel.setVisible(visible)
        self._info_divider.setVisible(visible)
        self.btn_info.setStyleSheet(INFO_ACTIVE_STYLE if visible else "")
        _save_settings({"info_panel_open": visible})

    # ── Folder loading ────────────────────────────────────────────────────────

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Scegli cartella foto")
        if not folder:
            return

        self._folder = Path(folder)
        self.photos  = sorted(
            p for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        self.selected.clear()
        self.current_index = -1

        _set_action("opening_folder", self._folder.name, len(self.photos))

        if not self.photos:
            self.image_view.show_placeholder("Nessuna foto trovata in questa cartella")
            self.status.showMessage("Nessuna foto trovata")
            self._sync_ui()
            return

        self.filmstrip.populate(len(self.photos))

        # Set up progress bar
        self.progress_bar.setMaximum(len(self.photos))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        if self._thumb_loader and self._thumb_loader.isRunning():
            self._thumb_loader.stop()
            self._thumb_loader.wait()

        self._thumb_loader = ThumbnailLoader(self.photos)
        self._thumb_loader.ready.connect(self.filmstrip.set_thumbnail)
        self._thumb_loader.progress.connect(self._on_thumb_progress)
        self._thumb_loader.start()

        self._go_to(0)
        self.setFocus()

    def _on_thumb_progress(self, n: int):
        self.progress_bar.setValue(n)
        if n >= self.progress_bar.maximum():
            self.progress_bar.hide()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if self.photos:
            self._go_to((self.current_index - 1) % len(self.photos))

    def _next(self):
        if self.photos:
            self._go_to((self.current_index + 1) % len(self.photos))

    def _go_to(self, index: int):
        if not self.photos or not (0 <= index < len(self.photos)):
            return
        self._current_location = ""
        self.current_index = index
        self.filmstrip.set_current(index)
        self._sync_ui()
        _set_action("navigating", self.photos[index].name)

        # Cancel any in-flight loaders from a previous navigation.
        if self._img_loader and self._img_loader.isRunning():
            self._img_loader.quit()
        if self._exif_loader and self._exif_loader.isRunning():
            self._exif_loader.quit()
        self.info_panel.clear()

        # Restart the debounce timer. If another _go_to fires within 150 ms
        # (e.g. held arrow key) the timer resets and no thread is started yet.
        self._nav_timer.start()

    def _load_current_image(self):
        """Called by _nav_timer after 150 ms of idle navigation."""
        index = self.current_index
        if not self.photos or not (0 <= index < len(self.photos)):
            return

        self._img_loader = ImageLoader(index, self.photos[index])
        self._img_loader.ready.connect(self._on_image_ready)
        self._img_loader.start()

        self._exif_loader = ExifLoader(index, self.photos[index])
        self._exif_loader.ready.connect(self._on_exif_ready)
        self._exif_loader.start()

        # Persist position so the next session (or post-crash reopen) resumes here.
        if self._folder:
            _save_settings({
                "last_folder": str(self._folder),
                "last_index":  index,
            })

    def _on_image_ready(self, index: int, px: QPixmap):
        if index != self.current_index:
            return
        if px.isNull():
            self.image_view.show_placeholder(
                f"Impossibile caricare: {self.photos[index].name}"
            )
        else:
            self.image_view.load(px)
            # If this load was triggered by a rotation, reuse the pixmap
            # to regenerate the thumbnail without a second file read.
            if self._rotate_thumb_idx == index:
                self._rotate_thumb_idx = None
                thumb = px.scaled(THUMB_W, THUMB_H,
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.filmstrip.set_thumbnail(index, thumb)
        self._update_watermark_preview()
        self._update_logo_preview()

    def _on_exif_ready(self, index: int, info: dict):
        if index != self.current_index:
            return
        self.info_panel.update_info(info)

        lat = info.get("gps_lat")
        lon = info.get("gps_lon")
        if lat is not None and lon is not None:
            if self._geo_loader and self._geo_loader.isRunning():
                self._geo_loader.quit()
            self._geo_loader = GeocoderThread(lat, lon)
            self._geo_loader.ready.connect(
                lambda loc, idx=index: self._on_geo_ready(idx, loc)
            )
            self._geo_loader.start()

    def _on_geo_ready(self, index: int, location: str):
        if index == self.current_index:
            self.info_panel.set_location(location)
            self._current_location = location
            self.lbl_name.setText(self._name_with_location())

    def _name_with_location(self) -> str:
        if self.current_index < 0:
            return ""
        name = self.photos[self.current_index].name
        if self._current_location:
            return f"{name}   ·   {self._current_location}"
        return name

    # ── Selection ─────────────────────────────────────────────────────────────

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _rotate(self, clockwise: bool):
        if self.current_index < 0:
            return
        if self._rotate_worker and self._rotate_worker.isRunning():
            return
        self.btn_rot_left.setEnabled(False)
        self.btn_rot_right.setEnabled(False)
        idx = self.current_index
        self._rotate_thumb_idx = idx
        self._rotate_worker = RotateWorker(idx, self.photos[idx], clockwise)
        self._rotate_worker.done.connect(self._on_rotate_done)
        self._rotate_worker.error.connect(self._on_rotate_error)
        self._rotate_worker.start()

    def _on_rotate_done(self, index: int):
        # Reload the image; _on_image_ready will also update the thumbnail
        if self._img_loader and self._img_loader.isRunning():
            self._img_loader.quit()
        self._img_loader = ImageLoader(index, self.photos[index])
        self._img_loader.ready.connect(self._on_image_ready)
        self._img_loader.start()
        # Also refresh EXIF panel (date/dimensions stay same but double-check)
        if self.info_panel.isVisible() and index == self.current_index:
            if self._exif_loader and self._exif_loader.isRunning():
                self._exif_loader.quit()
            self._exif_loader = ExifLoader(index, self.photos[index])
            self._exif_loader.ready.connect(self._on_exif_ready)
            self._exif_loader.start()
        self._sync_ui()

    def _on_rotate_error(self, msg: str):
        self._rotate_thumb_idx = None
        QMessageBox.warning(self, "Errore rotazione",
                            f"Impossibile ruotare la foto:\n{msg}")
        self._sync_ui()

    # ── Crop (saved as a new file, original untouched) ─────────────────────────

    def _crop_current(self):
        if self.current_index < 0 or self._folder is None:
            return
        idx  = self.current_index
        path = self.photos[idx]

        try:
            with PilImage.open(str(path)) as img:
                exif = img.getexif()
                # Apply EXIF orientation FIRST — exif_transpose reads it from
                # the same cached Exif object getexif() returned, so popping
                # the tag beforehand would silently leave the image un-rotated.
                img = PilImageOps.exif_transpose(img)
                exif.pop(274, None)          # now safe: pixels already corrected
                exif_bytes = exif.tobytes()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = _np.array(img)
            img_bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Impossibile aprire la foto:\n{e}")
            return

        dlg = CropDialog(
            str(path), self, img_bgr=img_bgr,
            title="Ritaglia foto",
            hint_text="Trascina per selezionare l'area da ritagliare.\n"
                      "Il ritaglio verrà salvato come nuovo file, senza modificare l'originale.",
        )
        if dlg.exec_() != QDialog.Accepted or dlg.crop_rect() is None:
            return

        crop = dlg.crop_rect()
        ih, iw = img_bgr.shape[:2]
        x  = max(0, min(crop.x(), iw - 1))
        y  = max(0, min(crop.y(), ih - 1))
        cw = min(crop.width(),  iw - x)
        ch = min(crop.height(), ih - y)
        if cw <= 0 or ch <= 0:
            return

        cropped_rgb = _cv2.cvtColor(img_bgr[y:y + ch, x:x + cw], _cv2.COLOR_BGR2RGB)
        cropped_img = PilImage.fromarray(cropped_rgb)

        new_path = path.parent / f"{path.stem}_crop{path.suffix}"
        counter = 2
        while new_path.exists():
            new_path = path.parent / f"{path.stem}_crop_{counter}{path.suffix}"
            counter += 1

        try:
            suffix = path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                cropped_img.save(str(new_path), format="JPEG", quality=95,
                                subsampling=0, exif=exif_bytes)
            elif suffix == ".png":
                cropped_img.save(str(new_path), format="PNG")
            else:
                cropped_img.save(str(new_path))
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Impossibile salvare il ritaglio:\n{e}")
            return

        insert_idx = idx + 1
        self.photos.insert(insert_idx, new_path)
        self.selected = {i + 1 if i >= insert_idx else i for i in self.selected}
        self.filmstrip.reindex_selected(lambda i: i + 1 if i >= insert_idx else i)

        self.filmstrip.blockSignals(True)
        new_item = QListWidgetItem()
        new_item.setSizeHint(QSize(THUMB_W + 8, THUMB_H + 8))
        self.filmstrip.insertItem(insert_idx, new_item)
        self.filmstrip.blockSignals(False)

        self._crop_loader = ThumbnailLoader([new_path])
        self._crop_loader.ready.connect(lambda _, px: self.filmstrip.set_thumbnail(insert_idx, px))
        self._crop_loader.start()

        self._go_to(insert_idx)
        self.status.showMessage(f"  Ritaglio salvato come '{new_path.name}'")

    def _toggle_select(self):
        if self.current_index < 0:
            return
        i = self.current_index
        if i in self.selected:
            self.selected.discard(i)
            self.filmstrip.set_selected(i, False)
        else:
            self.selected.add(i)
            self.filmstrip.set_selected(i, True)
        self._sync_ui()

    # ── Discard (move to Scartate) ────────────────────────────────────────────

    def _discard_current(self):
        self._tint_anim.stop()
        self.image_view.tintStrength = 0.0
        if self.current_index < 0 or self._folder is None:
            return

        idx = self.current_index
        src = self.photos[idx]

        _set_action("discarding_photo", src.name, len(self.photos))

        discard_dir = self._folder / DISCARD_FOLDER
        discard_dir.mkdir(exist_ok=True)

        dest = discard_dir / src.name
        if dest.exists():
            counter = 2
            while dest.exists():
                dest = discard_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Impossibile spostare la foto:\n{e}")
            return

        self._discard_history.append((src, dest, idx))

        # Update selected indices: remove idx, shift indices above it down by 1
        self.selected.discard(idx)
        self.selected = {i - 1 if i > idx else i for i in self.selected}
        self.filmstrip.reindex_selected(lambda i: None if i == idx else (i - 1 if i > idx else i))

        # Remove from filmstrip and photos list
        self.filmstrip.blockSignals(True)
        self.filmstrip.takeItem(idx)
        self.filmstrip.blockSignals(False)
        self.photos.pop(idx)

        if not self.photos:
            self.current_index = -1
            self.image_view.show_placeholder("Nessuna foto rimanente")
            self._sync_ui()
            self.status.showMessage(
                f"  '{src.name}' spostata in '{DISCARD_FOLDER}'  •  Ctrl+Z per annullare"
            )
            return

        # Stay on same position (or last photo if we were at the end)
        new_idx = min(idx, len(self.photos) - 1)
        self.current_index = -1  # force _go_to to reload
        self._go_to(new_idx)
        self.status.showMessage(
            f"  '{src.name}' spostata in '{DISCARD_FOLDER}'  •  Ctrl+Z per annullare"
        )

    def _undo_discard(self):
        if not self._discard_history:
            self.status.showMessage("  Nessuna azione da annullare")
            return

        original_path, discard_path, original_idx = self._discard_history.pop()

        if not discard_path.exists():
            QMessageBox.warning(
                self, "Annulla",
                f"Il file non esiste più:\n{discard_path.name}",
            )
            return

        dest = original_path
        if dest.exists():
            counter = 2
            while dest.exists():
                dest = original_path.parent / f"{original_path.stem}_{counter}{original_path.suffix}"
                counter += 1

        try:
            shutil.move(str(discard_path), str(dest))
        except Exception as e:
            QMessageBox.warning(self, "Errore annulla", f"Impossibile ripristinare la foto:\n{e}")
            self._discard_history.append((original_path, discard_path, original_idx))
            return

        insert_idx = min(original_idx, len(self.photos))
        self.photos.insert(insert_idx, dest)

        # Shift selected indices at or above insert_idx up by 1
        self.selected = {i + 1 if i >= insert_idx else i for i in self.selected}
        self.filmstrip.reindex_selected(lambda i: i + 1 if i >= insert_idx else i)

        # Insert placeholder item in filmstrip, then load its thumbnail
        self.filmstrip.blockSignals(True)
        new_item = QListWidgetItem()
        new_item.setSizeHint(QSize(THUMB_W + 8, THUMB_H + 8))
        self.filmstrip.insertItem(insert_idx, new_item)
        self.filmstrip.blockSignals(False)

        self._undo_loader = ThumbnailLoader([dest])
        self._undo_loader.ready.connect(lambda _, px: self.filmstrip.set_thumbnail(insert_idx, px))
        self._undo_loader.start()

        self._go_to(insert_idx)
        self.status.showMessage(f"  '{dest.name}' ripristinata")

    # ── Destination folder (session-scoped) ───────────────────────────────────

    def _set_dest_folder(self):
        name, ok = QInputDialog.getText(
            self, "Cartella di destinazione",
            "Nome cartella (solo per questa sessione):\n"
            f"Lascia vuoto per tornare al default  '{DEST_FOLDER}'",
            text=self._dest_folder,
        )
        if not ok:
            return
        self._dest_folder = name.strip() or DEST_FOLDER
        self._sync_ui()

    # ── Watermark ─────────────────────────────────────────────────────────────

    def _toggle_watermark(self):
        wms = _load_watermarks()
        if not wms["dark"] and not wms["light"]:
            self._watermark_setup()
            wms = _load_watermarks()
            if not wms["dark"] and not wms["light"]:
                return
        self._watermark_enabled = not self._watermark_enabled
        self._apply_watermark_btn_style()
        _save_settings({
            "watermark_enabled": self._watermark_enabled,
            "wm_override": self._wm_override or "off",
        })
        self._update_watermark_preview()

    def _apply_watermark_btn_style(self):
        suffix = {"dark": "  (scura)", "light": "  (chiara)"}.get(self._wm_override or "", "")
        self.btn_watermark.setText(f"✍  Firma{suffix}")
        self.btn_watermark.setStyleSheet(
            WATERMARK_ACTIVE_STYLE if self._watermark_enabled else ""
        )

    def _watermark_setup(self):
        wms = _load_watermarks()
        dlg = WatermarkSetupDialog(self, existing=wms)
        if dlg.exec_() != QDialog.Accepted:
            return
        WATERMARK_DIR.mkdir(exist_ok=True)
        for key in ("dark", "light"):
            src  = dlg.paths().get(key, "")
            dest = WATERMARK_DIR / f"{key}.png"
            if src:
                shutil.copy2(src, dest)
            elif dest.exists():
                dest.unlink()

    def _watermark_context_menu(self, pos):
        wms  = _load_watermarks()
        menu = QMenu(self)
        act_edit     = menu.addAction("✍  Modifica firma...")
        act_size_h   = menu.addAction("Dimensione firma orizzontale...")
        act_size_v   = menu.addAction("Dimensione firma verticale...")
        menu.addSeparator()
        act_col_auto  = menu.addAction("Colore: Automatico")
        act_col_dark  = menu.addAction("Colore: Scura")
        act_col_light = menu.addAction("Colore: Chiara")
        for act, key in ((act_col_auto, None), (act_col_dark, "dark"), (act_col_light, "light")):
            act.setCheckable(True)
            act.setChecked(self._wm_override == key)
        menu.addSeparator()
        act_rm_dark  = menu.addAction("Rimuovi firma scura/singola") if wms["dark"]  else None
        act_rm_light = menu.addAction("Rimuovi firma chiara")        if wms["light"] else None
        chosen = menu.exec_(self.btn_watermark.mapToGlobal(pos))
        if chosen == act_edit:
            self._watermark_setup()
            self._update_watermark_preview()
        elif chosen == act_size_h:
            self._watermark_resize("h")
        elif chosen == act_size_v:
            self._watermark_resize("v")
        elif chosen == act_col_auto:
            self._wm_override = None
            self._apply_watermark_btn_style()
            _save_settings({"wm_override": "off"})
            self._update_watermark_preview()
        elif chosen == act_col_dark:
            self._wm_override = "dark"
            self._apply_watermark_btn_style()
            _save_settings({"wm_override": "dark"})
            self._update_watermark_preview()
        elif chosen == act_col_light:
            self._wm_override = "light"
            self._apply_watermark_btn_style()
            _save_settings({"wm_override": "light"})
            self._update_watermark_preview()
        elif act_rm_dark and chosen == act_rm_dark:
            (WATERMARK_DIR / "dark.png").unlink(missing_ok=True)
            self._update_watermark_preview()
        elif act_rm_light and chosen == act_rm_light:
            (WATERMARK_DIR / "light.png").unlink(missing_ok=True)
            self._update_watermark_preview()

    def _watermark_resize(self, orientation: str):
        settings = _load_settings()
        if orientation == "h":
            key     = "watermark_size_pct_h"
            label   = "Larghezza firma — foto orizzontali (% della foto):"
            current = int(settings.get(key, 22))
        else:
            key     = "watermark_size_pct_v"
            label   = "Larghezza firma — foto verticali (% della foto):"
            current = int(settings.get(key, 22))
        val, ok = QInputDialog.getInt(self, "Dimensione firma", label, current, 5, 50, 1)
        if not ok:
            return
        _save_settings({key: val})
        self.image_view.set_watermark_sizes(
            _load_settings().get("watermark_size_pct_h", 22) / 100.0,
            _load_settings().get("watermark_size_pct_v", 22) / 100.0,
        )
        self._update_watermark_preview()

    def _cycle_wm_photo_color(self):
        """Cycles per-photo color override: (auto) → dark → light → (auto)."""
        if self.current_index < 0:
            return
        wms = _load_watermarks()
        if not (wms["dark"] and wms["light"]):
            return  # only one variant, nothing to cycle
        cur = self._wm_photo_override.get(self.current_index)
        if cur is None:
            # Determine auto choice, then set to the opposite
            img = _cv2.imread(str(self.photos[self.current_index]))
            if img is None:
                return
            auto = _pick_wm_path(img, wms)
            self._wm_photo_override[self.current_index] = (
                "light" if auto == wms["dark"] else "dark"
            )
        elif cur == "dark":
            self._wm_photo_override[self.current_index] = "light"
        else:
            del self._wm_photo_override[self.current_index]  # back to auto
        self._update_watermark_preview()

    def _update_watermark_preview(self):
        if (not self._watermark_enabled
                or self.current_index < 0
                or self.current_index not in self.selected):
            self.image_view.set_watermark_preview(None)
            return
        wms = _load_watermarks()
        if not wms["dark"] and not wms["light"]:
            self.image_view.set_watermark_preview(None)
            return
        path = self.photos[self.current_index]
        img  = _cv2.imread(str(path))
        if img is None:
            self.image_view.set_watermark_preview(None)
            return
        # Priority: per-photo override → global override → auto-detect
        per_photo = self._wm_photo_override.get(self.current_index)
        if per_photo == "dark":
            wm_path = wms["dark"] or wms["light"]
        elif per_photo == "light":
            wm_path = wms["light"] or wms["dark"]
        elif self._wm_override == "dark":
            wm_path = wms["dark"] or wms["light"]
        elif self._wm_override == "light":
            wm_path = wms["light"] or wms["dark"]
        else:
            wm_path = _pick_wm_path(img, wms)
        if wm_path is None:
            self.image_view.set_watermark_preview(None)
            return
        wm_px = QPixmap(str(wm_path))
        if wm_px.isNull():
            self.image_view.set_watermark_preview(None)
            return
        self.image_view.set_watermark_preview(wm_px)

    # ── Logo ──────────────────────────────────────────────────────────────────

    def _toggle_logo(self):
        if self.current_index < 0:
            return
        if self._logo_path is None:
            self._logo_setup()
            if self._logo_path is None:
                return
        if self.current_index in self.logo_marked:
            self.logo_marked.discard(self.current_index)
        else:
            self.logo_marked.add(self.current_index)
        self._apply_logo_btn_style()
        self._update_logo_preview()

    def _logo_setup(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona logo PNG", "", "PNG (*.png)"
        )
        if not path:
            return
        dlg = LogoPositionDialog(
            self,
            pos_h=self._logo_pos_h, pos_v=self._logo_pos_v,
            size_h=self._logo_size_h, size_v=self._logo_size_v,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._logo_path    = path
        raw = _np.fromfile(path, dtype=_np.uint8)
        self._logo_bgra    = _cv2.imdecode(raw, _cv2.IMREAD_UNCHANGED)
        self._logo_pos_h   = dlg.result_pos_h()
        self._logo_pos_v   = dlg.result_pos_v()
        self._logo_size_h  = dlg.result_size_h()
        self._logo_size_v  = dlg.result_size_v()

    def _logo_context_menu(self, pos):
        menu = QMenu(self)
        act_change = menu.addAction("🖼  Cambia logo...")
        act_pos    = menu.addAction("Cambia posizione...")
        chosen = menu.exec_(self.btn_logo.mapToGlobal(pos))
        if chosen == act_change:
            old_path = self._logo_path
            self._logo_path = None
            self._logo_setup()
            if self._logo_path is None:
                self._logo_path = old_path  # restore if cancelled
            self._update_logo_preview()
        elif chosen == act_pos and self._logo_path:
            dlg = LogoPositionDialog(
                self,
                pos_h=self._logo_pos_h, pos_v=self._logo_pos_v,
                size_h=self._logo_size_h, size_v=self._logo_size_v,
            )
            if dlg.exec_() == QDialog.Accepted:
                self._logo_pos_h  = dlg.result_pos_h()
                self._logo_pos_v  = dlg.result_pos_v()
                self._logo_size_h = dlg.result_size_h()
                self._logo_size_v = dlg.result_size_v()
                self._update_logo_preview()

    def _apply_logo_btn_style(self):
        is_marked = self.current_index in self.logo_marked
        self.btn_logo.setText("🖼  Logo  ✓" if is_marked else "🖼  Logo")
        self.btn_logo.setStyleSheet(LOGO_ACTIVE_STYLE if is_marked else "")

    def _update_logo_preview(self):
        if self.current_index < 0 or self.current_index not in self.logo_marked:
            self.image_view.set_logo_preview(None, "bot-right", 0.15)
            return
        if self._logo_path is None:
            self.image_view.set_logo_preview(None, "bot-right", 0.15)
            return
        px = self.image_view.current_pixmap()
        is_h    = px is not None and px.width() >= px.height()
        pos     = self._logo_pos_h  if is_h else self._logo_pos_v
        frac    = self._logo_size_h if is_h else self._logo_size_v
        logo_px = QPixmap(self._logo_path)
        if logo_px.isNull():
            self.image_view.set_logo_preview(None, "bot-right", 0.15)
            return
        self.image_view.set_logo_preview(logo_px, pos, frac / 100.0)

    # ── Copy ──────────────────────────────────────────────────────────────────

    def _copy_selected(self):
        if not self.selected or self._folder is None:
            return

        dest    = self._folder / self._dest_folder
        dest.mkdir(exist_ok=True)
        ordered = sorted(self.selected)

        progress = QProgressDialog(
            "Copio le foto selezionate...", "Annulla", 0, len(ordered), self
        )
        progress.setWindowTitle("Copia in corso")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumWidth(360)

        errors: list[str] = []
        copied = 0
        for i, idx in enumerate(ordered):
            if progress.wasCanceled():
                break
            src    = self.photos[idx]
            target = dest / src.name
            if target.exists():
                counter = 2
                while target.exists():
                    target = dest / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1
            try:
                shutil.copy2(src, target)
                if self._watermark_enabled:
                    try:
                        wms  = _load_watermarks()
                        img  = _cv2.imread(str(target))
                        if img is not None:
                            per_photo = self._wm_photo_override.get(idx)
                            if per_photo == "dark":
                                wm_path = wms["dark"] or wms["light"]
                            elif per_photo == "light":
                                wm_path = wms["light"] or wms["dark"]
                            elif self._wm_override == "dark":
                                wm_path = wms["dark"] or wms["light"]
                            elif self._wm_override == "light":
                                wm_path = wms["light"] or wms["dark"]
                            else:
                                wm_path = _pick_wm_path(img, wms)
                            if wm_path:
                                wm_bgra = _cv2.imread(str(wm_path), _cv2.IMREAD_UNCHANGED)
                                if wm_bgra is not None:
                                    ih, iw = img.shape[:2]
                                    frac   = (self.image_view._wm_size_frac_h
                                              if iw >= ih else
                                              self.image_view._wm_size_frac_v)
                                    result = _composite_watermark(img, wm_bgra, width_frac=frac)
                                    ext    = target.suffix.lower()
                                    params = ([_cv2.IMWRITE_JPEG_QUALITY, 95]
                                              if ext in ('.jpg', '.jpeg') else [])
                                    _cv2.imwrite(str(target), result, params)
                    except Exception:
                        pass
                if idx in self.logo_marked and self._logo_bgra is not None:
                    try:
                        img = _cv2.imread(str(target))
                        if img is not None:
                            ih2, iw2 = img.shape[:2]
                            is_h  = iw2 >= ih2
                            pos   = self._logo_pos_h  if is_h else self._logo_pos_v
                            frac  = (self._logo_size_h if is_h else self._logo_size_v) / 100.0
                            result = _composite_logo(img, self._logo_bgra, pos, frac)
                            ext    = target.suffix.lower()
                            params = ([_cv2.IMWRITE_JPEG_QUALITY, 95]
                                      if ext in ('.jpg', '.jpeg') else [])
                            _cv2.imwrite(str(target), result, params)
                    except Exception:
                        pass
                copied += 1
            except Exception as e:
                errors.append(f"{src.name}: {e}")
            progress.setValue(i + 1)

        progress.close()

        if errors:
            QMessageBox.warning(
                self, "Errori durante la copia",
                f"{copied} foto copiate.\n\nErrori:\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(
                self, "Copia completata",
                f"{copied} foto copiate in:\n{dest}",
            )

    # ── Filemail share ────────────────────────────────────────────────────────

    def _share_settings(self):
        """Open settings dialog directly (e.g. from right-click on btn_share)."""
        settings  = _load_settings()
        dlg = FilemailSettingsDialog(
            self,
            email=settings.get("filemail_email", ""),
            api_key=settings.get("filemail_apikey", ""),
        )
        if dlg.exec_() == QDialog.Accepted:
            _save_settings({"filemail_email": dlg.email(), "filemail_apikey": dlg.api_key()})

    def _share(self):
        if self._folder is None:
            return

        sel_folder = self._folder / self._dest_folder
        files = sorted(
            f for f in sel_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ) if sel_folder.exists() else []

        if not files:
            QMessageBox.information(
                self, "Nessuna foto da condividere",
                f"La cartella '{self._dest_folder}' è vuota o non esiste.\n"
                "Prima copia le foto selezionate con il pulsante  📋 Copia selezionate.",
            )
            return

        # ── 5 GB free-tier check ──────────────────────────────────────────────
        self._share_links   = []
        self._share_batch2  = []
        self._share_n_batches = 1

        total_bytes = sum(f.stat().st_size for f in files)
        if total_bytes > FILEMAIL_FREE_LIMIT:
            total_gb = total_bytes / (1024 ** 3)
            box = QMessageBox(self)
            box.setWindowTitle("Dimensione oltre il limite Filemail")
            box.setText(
                f"La cartella '{self._dest_folder}' pesa {total_gb:.1f} GB, "
                f"ma il piano gratuito di Filemail consente al massimo 5 GB per trasferimento.\n\n"
                "Cosa vuoi fare?"
            )
            btn_trim   = box.addButton("Invia solo i primi 5 GB",   QMessageBox.AcceptRole)
            btn_split  = box.addButton("Due trasferimenti (2 link)", QMessageBox.AcceptRole)
            btn_cancel = box.addButton("Annulla",                    QMessageBox.RejectRole)
            box.setDefaultButton(btn_cancel)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is btn_cancel or clicked is None:
                return

            # Build batch1 (up to 5 GB in order) and batch2 (remainder)
            batch1: list = [];  running = 0;  limit_hit = False
            batch2: list = []
            for f in files:
                sz = f.stat().st_size
                if not limit_hit and running + sz <= FILEMAIL_FREE_LIMIT:
                    batch1.append(f);  running += sz
                else:
                    limit_hit = True
                    if clicked is btn_split:
                        batch2.append(f)

            if not batch1:
                QMessageBox.warning(self, "Nessuna foto",
                                    "La prima foto supera già il limite di 5 GB.")
                return

            if clicked is btn_split and batch2:
                # Trim batch2 to 5 GB as well
                batch2_size = sum(f.stat().st_size for f in batch2)
                if batch2_size > FILEMAIL_FREE_LIMIT:
                    trimmed2: list = [];  r2 = 0
                    for f in batch2:
                        sz = f.stat().st_size
                        if r2 + sz <= FILEMAIL_FREE_LIMIT:
                            trimmed2.append(f);  r2 += sz
                        else:
                            break
                    batch2 = trimmed2
                    if batch2:
                        QMessageBox.information(
                            self, "Secondo trasferimento ridotto",
                            "Anche il secondo trasferimento supera 5 GB — "
                            "verranno inviati solo i file che rientrano nel limite."
                        )
                self._share_batch2  = batch2
                self._share_n_batches = 2

            files = batch1

        settings = _load_settings()
        email    = settings.get("filemail_email", "")
        api_key  = settings.get("filemail_apikey", "")

        if not email:
            dlg = FilemailSettingsDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            email   = dlg.email()
            api_key = dlg.api_key()
            _save_settings({"filemail_email": email, "filemail_apikey": api_key})

        self._share_email  = email
        self._share_apikey = api_key
        self._start_share_batch(files)

    def _start_share_batch(self, files: list):
        batch_num = len(self._share_links) + 1
        n = len(files)
        label = (
            f"Trasferimento {batch_num} di {self._share_n_batches} — "
            f"caricamento foto 0 di {n}..."
            if self._share_n_batches > 1
            else f"Caricamento foto 0 di {n}..."
        )
        self._share_progress = QProgressDialog(label, "Annulla", 0, n, self)
        self._share_progress.setWindowTitle("Caricamento su Filemail")
        self._share_progress.setWindowModality(Qt.WindowModal)
        self._share_progress.setMinimumWidth(400)
        self._share_progress.setValue(0)

        self._share_worker = FilemailUploader(files, self._share_email, self._share_apikey)
        self._share_worker.progress.connect(self._on_share_progress)
        self._share_worker.done.connect(self._on_share_done)
        self._share_worker.warning.connect(self._on_share_warning)
        self._share_worker.error.connect(self._on_share_error)
        self._share_progress.canceled.connect(self._share_worker.stop)
        self._share_worker.start()
        self._share_progress.show()

    def _on_share_progress(self, done: int, total: int):
        if self._share_progress:
            self._share_progress.setValue(done)
            batch_num = len(self._share_links) + 1
            prefix = (
                f"Trasferimento {batch_num} di {self._share_n_batches} — "
                if self._share_n_batches > 1 else ""
            )
            self._share_progress.setLabelText(
                f"{prefix}Caricamento foto {done} di {total}..."
            )

    def _on_share_done(self, url: str):
        if self._share_progress:
            self._share_progress.close()
            self._share_progress = None
        self._share_links.append(url)

        if self._share_batch2:
            # Launch the second transfer
            batch2 = self._share_batch2
            self._share_batch2 = []
            self._start_share_batch(batch2)
            return

        # All batches complete — show results
        if len(self._share_links) == 1:
            ShareLinkDialog(self, self._share_links[0]).exec_()
        else:
            self._show_multi_link_dialog(self._share_links)
        self._share_links = []

    def _show_multi_link_dialog(self, urls: list[str]):
        dlg = QDialog(self)
        dlg.setWindowTitle("Foto condivise!")
        dlg.setMinimumWidth(540)
        dlg.setStyleSheet(BASE_STYLE)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(24, 24, 24, 24)

        title = QLabel(f"✅  Foto caricate in {len(urls)} trasferimenti!")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00c870;")
        lay.addWidget(title)

        sub = QLabel(
            "Le foto sono state divise in due link (disponibili 7 giorni ciascuno).\n"
            "Condividi entrambi i link con il destinatario."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #aaa;")
        lay.addWidget(sub)

        copy_btns: list[QPushButton] = []
        for i, url in enumerate(urls):
            lbl = QLabel(f"Link {i + 1}:")
            lbl.setStyleSheet("color: #ccc; font-weight: bold;")
            lay.addWidget(lbl)

            row = QHBoxLayout()
            edit = QLineEdit(url)
            edit.setReadOnly(True)
            edit.setStyleSheet(
                "background:#1a1a1a; color:#f0f0f0; padding:6px;"
                "border:1px solid #444; border-radius:4px;"
            )
            row.addWidget(edit, 1)

            btn_copy = QPushButton("📋  Copia")
            btn_copy.setFixedHeight(34)
            copy_btns.append(btn_copy)

            def _copy(u=url, b=btn_copy):
                QApplication.clipboard().setText(u)
                b.setText("✅  Copiato!")
            btn_copy.clicked.connect(_copy)
            row.addWidget(btn_copy)
            lay.addLayout(row)

            btn_browser = QPushButton(f"🌐  Apri link {i + 1} nel browser")
            btn_browser.clicked.connect(lambda _, u=url: QDesktopServices.openUrl(QUrl(u)))
            lay.addWidget(btn_browser)

        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet(ACCENT_STYLE)
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)
        dlg.exec_()

    def _on_share_warning(self, msg: str):
        QMessageBox.warning(self, "Alcune foto non caricate", msg)

    def _on_share_error(self, msg: str):
        if self._share_progress:
            self._share_progress.close()
            self._share_progress = None
        QMessageBox.critical(
            self, "Errore Filemail",
            f"Impossibile completare il caricamento:\n\n{msg}\n\n"
            "Verifica la connessione e le impostazioni email\n"
            "(tasto destro su 📤 Condividi → Impostazioni).",
        )

    # ── Sync UI state ─────────────────────────────────────────────────────────

    def _sync_ui(self):
        has   = bool(self.photos)
        cur   = self.current_index >= 0
        n_sel = len(self.selected)

        for btn in (self.btn_prev, self.btn_next,
                    self.btn_img_prev, self.btn_img_next):
            btn.setEnabled(has)
        self.btn_select.setEnabled(cur)
        self.btn_watermark.setEnabled(cur)
        self.btn_logo.setEnabled(cur)
        self.btn_info.setEnabled(cur)
        self.btn_copy.setEnabled(n_sel > 0)
        self.btn_share.setEnabled(self._folder is not None)
        rotating = bool(self._rotate_worker and self._rotate_worker.isRunning())
        for btn in (self.btn_rot_left, self.btn_rot_right):
            btn.setEnabled(cur and not rotating)
        self.btn_crop.setEnabled(cur and not rotating)
        for btn in (self.btn_zin, self.btn_zout, self.btn_zreset):
            btn.setEnabled(cur)

        if cur:
            self.lbl_counter.setText(f"{self.current_index + 1} / {len(self.photos)}")
            self.lbl_name.setText(self._name_with_location())
            is_sel = self.current_index in self.selected
            self.btn_select.setText(
                "★  Selezionata  [Space]" if is_sel else "☆  Seleziona  [Space]"
            )
            self.btn_select.setStyleSheet(SELECT_ACTIVE_STYLE if is_sel else "")
            self._apply_watermark_btn_style()
            self._apply_logo_btn_style()
            self._update_watermark_preview()
            if is_sel:
                self.image_view.start_snake()
            else:
                self.image_view.stop_snake()
        else:
            self.lbl_counter.setText("—")
            self.lbl_name.setText("")
            self.btn_select.setText("☆  Seleziona  [Space]")
            self.btn_select.setStyleSheet("")
            self.image_view.stop_snake()

        folder_label = (
            f"→ {self._dest_folder}"
            if self._dest_folder != DEST_FOLDER else ""
        )
        self.btn_copy.setText(
            f"📋  Copia{' ' + folder_label if folder_label else ''}  ({n_sel})"
            if n_sel else
            f"📋  Copia selezionate{' ' + folder_label if folder_label else ''}"
        )
        if n_sel > 0 and self.photos:
            total = sum(
                self.photos[i].stat().st_size
                for i in self.selected if 0 <= i < len(self.photos)
            )
            self.btn_copy.setToolTip(
                f"Dimensione totale: {_fmt_size(total)}\n"
                "Tasto destro → cartella di destinazione"
            )
        else:
            self.btn_copy.setToolTip("Tasto destro → cartella di destinazione")

        if cur:
            sel_info = f"{n_sel} selezionate" if n_sel else "Nessuna selezionata"
            self.status.showMessage(
                f"  {self.photos[self.current_index].name}"
                f"   •   {self.current_index + 1} di {len(self.photos)}"
                f"   •   {sel_info}"
            )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_previous_session_crash()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon = QIcon(resource_path("icon.ico"))
    app.setWindowIcon(icon)

    window = PhotoSelector()
    window.setWindowIcon(icon)
    window.showMaximized()
    sys.exit(app.exec_())
