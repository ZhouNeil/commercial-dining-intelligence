"""
Yelp Photos bundle under `data/Yelp Photos/`:
  - `photos.json` (or `yelp_photos/photos.json`) — JSONL: photo_id, business_id, caption, label
  - optional `yelp_photos.tar` — contains `photos/{photo_id}.jpg` (not required if images are on disk)
  - extracted JPGs: either `photos/{id}.jpg` or `yelp_photos/photos/{id}.jpg` (nested tar layout)
"""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Optional


def yelp_photos_bundle_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "Yelp Photos"


def photos_json_path(base_dir: Path) -> Path:
    """Preferred path; file may not exist — prefer `resolve_photos_json`."""
    return yelp_photos_bundle_dir(base_dir) / "photos.json"


def resolve_photos_json(base_dir: Path) -> Optional[Path]:
    """First existing JSONL manifest under the bundle directory."""
    b = yelp_photos_bundle_dir(base_dir)
    for cand in (b / "photos.json", b / "yelp_photos" / "photos.json"):
        if cand.is_file():
            return cand
    return None


def photos_tar_path(base_dir: Path) -> Path:
    return yelp_photos_bundle_dir(base_dir) / "yelp_photos.tar"


def extracted_photos_dir(base_dir: Path) -> Path:
    return yelp_photos_bundle_dir(base_dir) / "photos"


def nested_extracted_photos_dir(base_dir: Path) -> Path:
    """Layout when the tar root was extracted as `yelp_photos/photos/`."""
    return yelp_photos_bundle_dir(base_dir) / "yelp_photos" / "photos"


def _local_jpg_candidates(base_dir: Path, photo_id: str) -> list[Path]:
    name = f"{photo_id}.jpg"
    bundle = yelp_photos_bundle_dir(base_dir)
    return [
        bundle / "photos" / name,
        nested_extracted_photos_dir(base_dir) / name,
    ]


def load_business_photo_ids(
    photos_json: Optional[Path],
    *,
    max_per_business: int = 10,
) -> dict[str, list[str]]:
    """Map business_id -> up to max_per_business photo_id strings (first seen in file order)."""
    if photos_json is None or not photos_json.is_file():
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    with photos_json.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = o.get("business_id")
            pid = o.get("photo_id")
            if not bid or not pid:
                continue
            bid_s = str(bid)
            if len(out[bid_s]) >= max_per_business:
                continue
            out[bid_s].append(str(pid))
    return dict(out)


class YelpPhotoTarReader:
    """Keep one tar handle for repeated extractfile (Streamlit @cache_resource)."""

    def __init__(self, tar_path: Path):
        self.tar_path = Path(tar_path)
        self._tf: Optional[tarfile.TarFile] = None

    def _ensure(self) -> None:
        if self._tf is None:
            self._tf = tarfile.open(self.tar_path, "r:*")

    def read_jpg(self, photo_id: str) -> Optional[bytes]:
        name = f"photos/{photo_id}.jpg"
        self._ensure()
        assert self._tf is not None
        try:
            m = self._tf.getmember(name)
        except KeyError:
            return None
        fh = self._tf.extractfile(m)
        if fh is None:
            return None
        try:
            return fh.read()
        finally:
            fh.close()

    def __del__(self) -> None:
        if self._tf is not None:
            try:
                self._tf.close()
            except Exception:
                pass


def read_photo_jpg_bytes(
    base_dir: Path,
    photo_id: str,
    archive: Optional[YelpPhotoTarReader],
) -> Optional[bytes]:
    """Prefer on-disk JPGs (`photos/` or `yelp_photos/photos/`); else read from tar if provided."""
    for disk in _local_jpg_candidates(base_dir, photo_id):
        if disk.is_file():
            return disk.read_bytes()
    if archive is not None:
        return archive.read_jpg(photo_id)
    return None


def has_local_photo_folder(base_dir: Path) -> bool:
    """True if any expected folder exists and contains at least one `.jpg`."""
    for d in (extracted_photos_dir(base_dir), nested_extracted_photos_dir(base_dir)):
        if d.is_dir() and any(d.glob("*.jpg")):
            return True
    return False
