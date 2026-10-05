"""国土地理院シームレス空中写真タイルの取得・キャッシュ・屋根クロップ。"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from .config import TILE_URL, TILE_ZOOM, USER_AGENT

TILE = 256


def lonlat_to_pixel(lon, lat, z: int = TILE_ZOOM):
    """経緯度 -> Web メルカトルのグローバルピクセル座標。"""
    n = TILE * 2 ** z
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    s = np.sin(np.radians(lat))
    x = (lon + 180.0) / 360.0 * n
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def meters_per_pixel(lat: float, z: int = TILE_ZOOM) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** z


class TileStore:
    """ディスクキャッシュ付きタイル取得。offline=True ならネットワークに出ない。"""

    def __init__(self, root: Path, zoom: int = TILE_ZOOM, url: str = TILE_URL,
                 offline: bool = False, delay: float = 0.0, ext: str = "jpg"):
        self.root, self.zoom, self.url, self.ext = Path(root), zoom, url, ext
        self.offline, self.delay = offline, delay
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._get = lru_cache(maxsize=512)(self._load)

    def path(self, x: int, y: int) -> Path:
        return self.root / str(self.zoom) / str(x) / f"{y}.{self.ext}"

    def _load(self, x: int, y: int) -> np.ndarray | None:
        p = self.path(x, y)
        if not p.exists():
            if self.offline or not self._fetch(x, y):
                return None
        return np.asarray(Image.open(p).convert("RGB"))

    def _fetch(self, x: int, y: int) -> bool:
        p = self.path(x, y)
        for attempt in range(3):
            try:
                r = self._session.get(self.url.format(z=self.zoom, x=x, y=y), timeout=30)
                if r.status_code == 404:
                    return False  # 撮影範囲外
                r.raise_for_status()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
                if self.delay:
                    time.sleep(self.delay)
                return True
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return False

    def tile(self, x: int, y: int) -> np.ndarray | None:
        return self._get(x, y)

    def prefetch(self, tiles: set[tuple[int, int]], workers: int = 4) -> tuple[int, int]:
        """未取得タイルをダウンロード。(新規取得数, 失敗/範囲外数) を返す。"""
        todo = [t for t in tiles if not self.path(*t).exists()]
        if self.offline:
            return 0, len(todo)
        with ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(lambda t: self._fetch(*t), todo))
        return sum(ok), len(ok) - sum(ok)


@dataclass
class Crop:
    img: np.ndarray        # H x W x 3 uint8（建物外接矩形 + 余白）
    mask: np.ndarray       # H x W bool（建物ポリゴン内）
    poly_px: list          # クロップ内のポリゴン頂点 [(x, y), ...]
    missing: int           # 取得できなかったタイル数


def _px_bounds(geom, z, margin):
    minx, miny, maxx, maxy = geom.bounds
    x0, y1 = lonlat_to_pixel(minx, miny, z)   # 北が y 小
    x1, y0 = lonlat_to_pixel(maxx, maxy, z)
    return (int(math.floor(x0)) - margin, int(math.floor(y0)) - margin,
            int(math.ceil(x1)) + margin, int(math.ceil(y1)) + margin)


def needed_tiles(geoms, z: int = TILE_ZOOM, margin: int = 12) -> set[tuple[int, int]]:
    out = set()
    for g in geoms:
        x0, y0, x1, y1 = _px_bounds(g, z, margin)
        for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
            for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
                out.add((tx, ty))
    return out


def building_crop(geom, store: TileStore, margin: int = 12) -> Crop:
    z = store.zoom
    x0, y0, x1, y1 = _px_bounds(geom, z, margin)
    tx0, ty0 = x0 // TILE, y0 // TILE
    tx1, ty1 = (x1 - 1) // TILE, (y1 - 1) // TILE
    canvas = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    missing = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = store.tile(tx, ty)
            if t is None:
                missing += 1
                continue
            canvas[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = t
    ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
    img = canvas[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
    xs, ys = lonlat_to_pixel(*zip(*geom.exterior.coords), z)
    poly = [(float(a - x0), float(b - y0)) for a, b in zip(xs, ys)]
    m = Image.new("L", (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255)
    return Crop(img=img, mask=np.asarray(m) > 0, poly_px=poly, missing=missing)
