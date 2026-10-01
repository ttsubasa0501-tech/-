"""国土地理院 最適化ベクトルタイル(基盤地図情報由来の BldA)から建物外周を取得。

基盤地図情報のダウンロードはログインが必要なため、GML が無い場合の代替ソース。
地物コードは基盤地図情報の BldA と同じ(3101 普通建物, 3102 堅ろう建物, 3111/3112 無壁舎)。
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile as mvt
import numpy as np
import requests
from shapely.geometry import Polygon, box

from .config import CRS_GEO, USER_AGENT

URL = "https://cyberjapandata.gsi.go.jp/xyz/optimal_bvmap-v1/{z}/{x}/{y}.pbf"
ZOOM = 16            # 建物が入る最大ズーム
EXTENT = 4096
BUFFER = 80          # タイル余白（座標は -80..4176）
TYPES = {3101: "普通建物", 3102: "堅ろう建物", 3111: "普通無壁舎", 3112: "堅ろう無壁舎"}


def _tiles_for(boundary: gpd.GeoDataFrame, z: int) -> list[tuple[int, int]]:
    n = 2 ** z
    out = set()
    for g in boundary.geometry:
        minx, miny, maxx, maxy = g.bounds
        x0, x1 = (int((v + 180) / 360 * n) for v in (minx, maxx))
        y0 = int((1 - math.asinh(math.tan(math.radians(maxy))) / math.pi) / 2 * n)
        y1 = int((1 - math.asinh(math.tan(math.radians(miny))) / math.pi) / 2 * n)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                out.add((x, y))
    return sorted(out)


def _fetch(args) -> tuple[tuple[int, int], bytes | None]:
    (x, y), cache, z = args
    p = cache / str(z) / str(x) / f"{y}.pbf"
    if p.exists():
        return (x, y), p.read_bytes()
    for _ in range(3):
        try:
            r = requests.get(URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 404:
                return (x, y), None
            r.raise_for_status()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
            return (x, y), r.content
        except requests.RequestException:
            continue
    return (x, y), None


def _to_lonlat(tx, ty, px, py, z):
    n = EXTENT * 2 ** z
    gx, gy = tx * EXTENT + px, ty * EXTENT + py
    lon = gx / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * gy / n))))
    return lon, lat


def load_bldA_vtile(boundary: gpd.GeoDataFrame, cache: Path, z: int = ZOOM,
                    workers: int = 4) -> gpd.GeoDataFrame:
    """境界(町丁目)の範囲を覆うタイルから BldA を読み、EPSG:6668 の GeoDataFrame にする。

    隣接タイルの余白で重複するため、「重心がタイル本体内」かつ「余白端で切れていない」建物だけ採用する。
    """
    tiles = _tiles_for(boundary.to_crs(CRS_GEO), z)
    print(f"[vtile] z{z} タイル {len(tiles)} 枚")
    with ThreadPoolExecutor(workers) as ex:
        data = list(ex.map(_fetch, [(t, cache, z) for t in tiles]))
    lo, hi = -BUFFER + 1, EXTENT + BUFFER - 1
    rows, clipped = [], 0
    for (tx, ty), blob in data:
        if not blob:
            continue
        layer = mvt.decode(blob, default_options={"y_coord_down": True}).get("BldA")
        if not layer:
            continue
        for f in layer["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for rings in polys:
                a = np.asarray(rings[0], float)
                if len(a) < 4:
                    continue
                if a.min() < lo or a.max() > hi:
                    clipped += 1           # 余白端で切れた断片（隣タイルに完全形がある）
                    continue
                cx, cy = a[:, 0].mean(), a[:, 1].mean()
                if not (0 <= cx < EXTENT and 0 <= cy < EXTENT):
                    continue               # 他タイルが本体として持つ
                ring_ll = lambda r: list(zip(*_to_lonlat(tx, ty, np.asarray(r, float)[:, 0],
                                                         np.asarray(r, float)[:, 1], z)))
                try:
                    poly = Polygon(ring_ll(rings[0]), [ring_ll(h) for h in rings[1:] if len(h) >= 4])
                except ValueError:
                    continue
                rows.append({"fgd_fid": None, "bld_type": TYPES.get(f["properties"].get("vt_code")),
                             "geometry": poly})
    print(f"[vtile] 建物 {len(rows)} 棟（余白端で切れた断片 {clipped} 件は除外）")
    if not rows:
        raise ValueError("建物が取得できませんでした")
    return gpd.GeoDataFrame(rows, crs=CRS_GEO)
