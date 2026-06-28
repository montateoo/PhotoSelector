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
    QDialog, QLineEdit, QDialogButtonBox, QInputDialog,
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
FILEMAIL_CHUNK = 10 * 1024 * 1024          # 10 MB per chunk
SETTINGS_PATH  = Path.home() / ".photoselector_settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

logging.basicConfig(
    filename=str(_APP_LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("PhotoSelector")

_session_state = {"action": "starting", "detail": "", "photos": 0}


def _set_action(action: str, detail: str = "", photos: int | None = None) -> None:
    """Records a breadcrumb of what the app was doing, for crash context."""
    _session_state["action"] = action
    _session_state["detail"] = detail
    if photos is not None:
        _session_state["photos"] = photos
    try:
        _STATE_PATH.write_text(json.dumps({
            **_session_state, "clean_exit": False, "ts": time.time(),
        }), encoding="utf-8")
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
        n = len(self.photos)
        for i, path in enumerate(self.photos):
            if not self._active:
                break
            if i % 25 == 0:
                _set_action("generating_thumbnails", f"{i}/{n}", n)
            try:
                px = _load_oriented_pixmap(path, max_size=(THUMB_W * 2, THUMB_H * 2))
                if not px.isNull():
                    thumb = px.scaled(THUMB_W, THUMB_H,
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
                    self.ready.emit(i, thumb)
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

            t      = data.get("transfer", data)
            tid    = t["transferid"]
            tkey   = t["transferkey"]
            turl   = t["transferurl"]

            # 2. Upload files one by one (chunked for large files)
            for i, path in enumerate(self._files):
                if self._stopped:
                    return
                self._upload_file(Path(path), turl, tid, tkey)
                self.progress.emit(i + 1, len(self._files))

            # 3. Complete transfer
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

        except Exception as exc:
            self.error.emit(str(exc))

    def _upload_file(self, path: Path, turl: str, tid: str, tkey: str):
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
                req = urllib.request.Request(
                    f"{turl}?{qs}",
                    data=chunk,
                    method="POST",
                    headers={
                        "Content-Type":   "application/octet-stream",
                        "Content-Length": str(len(chunk)),
                        "User-Agent":     "PhotoSelector/1.0",
                    },
                )
                self._send_chunk_with_retry(req, path.name, offset)
                offset += len(chunk)

        # Defensive check: a silently-truncated upload must not be reported as success
        if offset != size:
            raise RuntimeError(
                f"Caricamento incompleto per {path.name}: "
                f"inviati {offset} byte su {size}"
            )

    def _send_chunk_with_retry(self, req, filename: str, offset: int, attempts: int = 3):
        """Uploads one chunk, validating the response and retrying transient failures.

        Filemail's chunk endpoint can return HTTP 200 with an app-level error in
        the JSON body; previously that body was discarded unread, so a failed
        chunk silently passed and the file ended up corrupted on the recipient
        side with no error shown to the user.
        """
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            if self._stopped:
                return
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
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
                    time.sleep(1.5 * attempt)
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

# ── Crop selection widget + dialog ────────────────────────────────────────────

class _CropCanvas(QWidget):
    """Image widget that lets the user drag a crop rectangle."""
    crop_changed = pyqtSignal(QRect)

    def __init__(self, qimg: 'QImage', parent=None):
        super().__init__(parent)
        self._img      = qimg
        self._start:   QPoint | None = None
        self._cur:     QPoint | None = None
        self._dragging = False
        self.setCursor(Qt.CrossCursor)

    def clear_selection(self):
        self._start = self._cur = None
        self._dragging = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._start    = e.pos()
            self._cur      = e.pos()
            self._dragging = True
            self.update()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._cur = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._dragging:
            self._cur      = e.pos()
            self._dragging = False
            r = self._sel()
            if r is not None and r.width() > 8 and r.height() > 8:
                self.crop_changed.emit(r)
            self.update()

    def _sel(self) -> 'QRect | None':
        if self._start is None or self._cur is None:
            return None
        return QRect(self._start, self._cur).normalized()

    def paintEvent(self, e):
        p = QPainter(self)
        p.drawImage(0, 0, self._img)
        r = self._sel()
        if r:
            w, h = self.width(), self.height()
            dim = QColor(0, 0, 0, 130)
            p.fillRect(0,        0,        w,        r.top(),              dim)
            p.fillRect(0,        r.bottom(), w,       h - r.bottom(),      dim)
            p.fillRect(0,        r.top(),   r.left(), r.height(),           dim)
            p.fillRect(r.right(), r.top(),  w - r.right(), r.height(),      dim)
            p.setPen(QPen(QColor(255, 70, 70), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            # corner handles
            p.setBrush(QColor(255, 70, 70))
            for pt in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                p.drawEllipse(pt, 4, 4)


class CropDialog(QDialog):
    """Shows a photo and lets the user drag a crop selection."""

    def __init__(self, image_path: str, parent=None, img_bgr: '_np.ndarray | None' = None,
                 title: str = "Ritaglia foto",
                 hint_text: str = "Trascina per selezionare l'area da ritagliare.\n"
                                  "Lascia senza selezione per usare la foto intera."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        if img_bgr is not None:
            img = img_bgr
        else:
            img = _cv2.imread(image_path)
            if img is None:
                img = _np.zeros((100, 100, 3), dtype=_np.uint8)
        oh, ow = img.shape[:2]
        screen = QApplication.primaryScreen()
        avail  = screen.availableGeometry() if screen else None
        max_w  = int(avail.width()  * 0.85) if avail else 1000
        max_h  = int(avail.height() * 0.80) if avail else 750
        self._scale = min(max_w / ow, max_h / oh, 1.0)
        dw, dh = int(ow * self._scale), int(oh * self._scale)

        rgb   = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, ow, oh, 3 * ow, QImage.Format_RGB888).scaled(
            dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._crop: QRect | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        hint = QLabel(hint_text)
        hint.setStyleSheet("color:#aaa; font-size:12px;")
        lay.addWidget(hint)

        self._canvas = _CropCanvas(qimg, self)
        self._canvas.setFixedSize(dw, dh)
        self._canvas.crop_changed.connect(self._on_crop)
        lay.addWidget(self._canvas, alignment=Qt.AlignCenter)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btn_clear = btns.addButton("Usa foto intera", QDialogButtonBox.ResetRole)
        btn_clear.clicked.connect(self._clear)
        btn_clear.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(btns)

        self.adjustSize()

    def _on_crop(self, rect: QRect):
        s = 1.0 / self._scale
        self._crop = QRect(int(rect.x() * s), int(rect.y() * s),
                           int(rect.width() * s), int(rect.height() * s))

    def _clear(self):
        self._crop = None
        self._canvas.clear_selection()

    def crop_rect(self) -> 'QRect | None':
        return self._crop


# ── Image view ────────────────────────────────────────────────────────────────

class ImageView(QGraphicsView):
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

    # ── Overlay drawing (tint + snake) ────────────────────────────────────────

    def drawForeground(self, painter, rect):
        has_tint  = self._tint_strength > 0.0
        has_snake = self._snake_active
        if not has_tint and not has_snake:
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

    def mousePressEvent(self, event):
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
        self._rotate_worker:  RotateWorker | None      = None
        self._rotate_thumb_idx: int | None             = None  # needs thumb regen
        self._share_worker:   FilemailUploader | None  = None
        self._share_progress: QProgressDialog | None   = None
        self._dest_folder:    str                       = DEST_FOLDER
        self._discard_history: list[tuple[Path, Path, int]] = []
        self._space_press_idx: int = -1
        self._current_location: str = ""

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

        for w in [self.btn_open,
                  _vsep(), self.btn_prev, self.lbl_counter, self.btn_next,
                  _vsep(), self.lbl_name,
                  _vsep(), self.btn_select, self.btn_copy, self.btn_share,
                  _vsep(), self.btn_info,
                  _vsep(), self.btn_rot_left, self.btn_rot_right, self.btn_crop,
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

        # Load full image
        if self._img_loader and self._img_loader.isRunning():
            self._img_loader.quit()
        self._img_loader = ImageLoader(index, self.photos[index])
        self._img_loader.ready.connect(self._on_image_ready)
        self._img_loader.start()

        # Load EXIF (always, so info panel is ready when opened)
        if self._exif_loader and self._exif_loader.isRunning():
            self._exif_loader.quit()
        self.info_panel.clear()
        self._exif_loader = ExifLoader(index, self.photos[index])
        self._exif_loader.ready.connect(self._on_exif_ready)
        self._exif_loader.start()

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

        n = len(files)
        self._share_progress = QProgressDialog(
            f"Caricamento foto 0 di {n}...", "Annulla", 0, n, self
        )
        self._share_progress.setWindowTitle("Caricamento su Filemail")
        self._share_progress.setWindowModality(Qt.WindowModal)
        self._share_progress.setMinimumWidth(400)
        self._share_progress.setValue(0)

        self._share_worker = FilemailUploader(files, email, api_key)
        self._share_worker.progress.connect(self._on_share_progress)
        self._share_worker.done.connect(self._on_share_done)
        self._share_worker.error.connect(self._on_share_error)
        self._share_progress.canceled.connect(self._share_worker.stop)
        self._share_worker.start()
        self._share_progress.show()

    def _on_share_progress(self, done: int, total: int):
        if self._share_progress:
            self._share_progress.setValue(done)
            self._share_progress.setLabelText(
                f"Caricamento foto {done} di {total}..."
            )

    def _on_share_done(self, url: str):
        if self._share_progress:
            self._share_progress.close()
            self._share_progress = None
        ShareLinkDialog(self, url).exec_()

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
    window.show()
    sys.exit(app.exec_())
