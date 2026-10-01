"""オフライン検証用の合成データ(境界・BldA GML・空中写真タイル・世帯数CSV)を作る。"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box

from solarmap.tiles import TILE, lonlat_to_pixel

LON0, LAT0 = 141.240, 43.120
DLON, DLAT = 0.006, 0.004
M_LON, M_LAT = 1.2327e-5, 9.0e-6   # 1 m あたりの度（北緯43度付近）
PANEL_PCT = {"A": 0.0, "B": 0.1, "C": 0.3, "D": 0.5}   # 町丁目ごとの設置率(仕込み)
DETACHED = {"A": 200, "B": 200, "C": 200, "D": 200}


def _rect(cx, cy, w, h):  # 中心(lon,lat)・幅高さ(m) -> 角の (lon,lat)
    return [(cx - w / 2 * M_LON, cy - h / 2 * M_LAT), (cx + w / 2 * M_LON, cy - h / 2 * M_LAT),
            (cx + w / 2 * M_LON, cy + h / 2 * M_LAT), (cx - w / 2 * M_LON, cy + h / 2 * M_LAT)]


def make(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    towns = {}
    for i, k in enumerate("ABCD"):
        x0, y0 = LON0 + (i % 2) * DLON, LAT0 + (i // 2) * DLAT
        towns[k] = (x0, y0)
    boundary = gpd.GeoDataFrame(
        {"KEY_CODE": [f"0111000{ord(k)-64:02d}" for k in towns], "S_NAME": [f"{k}町丁目" for k in towns],
         "geometry": [box(x, y, x + DLON, y + DLAT) for x, y in towns.values()]}, crs="EPSG:4612")
    boundary.to_file(root / "boundary.gpkg", driver="GPKG")

    blds = []  # (corners, town, is_panel, kind)
    for k, (x0, y0) in towns.items():
        n = 0
        for r in range(2, 14):
            for c in range(2, 16):
                cx, cy = x0 + (c * 30) * M_LON, y0 + (r * 30) * M_LAT
                kind = "house"
                w, h = 8, 10
                if (r, c) == (3, 3):
                    kind, w, h = "shed", 4, 5       # 20㎡: 除外されるはず
                elif (r, c) == (3, 5):
                    kind, w, h = "big", 25, 20      # 500㎡: 除外
                panel = kind == "house" and (n % 10) < PANEL_PCT[k] * 10
                n += kind == "house"
                blds.append((_rect(cx, cy, w, h), k, panel, kind))

    # --- 空中写真（グローバルピクセル座標で描画 -> タイル分割）
    z = 18
    gx0, gy1 = lonlat_to_pixel(LON0 - 0.001, LAT0 - 0.001, z)
    gx1, gy0 = lonlat_to_pixel(LON0 + 2 * DLON + 0.001, LAT0 + 2 * DLAT + 0.001, z)
    tx0, ty0, tx1, ty1 = int(gx0 // TILE), int(gy0 // TILE), int(gx1 // TILE), int(gy1 // TILE)
    W, H = (tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal([95, 120, 80], 10, (H, W, 3)), 0, 255).astype(np.uint8)  # 草地
    im = Image.fromarray(base)
    d = ImageDraw.Draw(im)
    ox, oy = tx0 * TILE, ty0 * TILE

    def px(c):
        xs, ys = lonlat_to_pixel([p[0] for p in c], [p[1] for p in c], z)
        return [(float(a - ox), float(b - oy)) for a, b in zip(xs, ys)]

    for corners, k, panel, kind in blds:
        p = px(corners)
        d.polygon(p, fill=(150, 125, 105))             # 茶色の屋根
        if panel:
            xs, ys = zip(*p)
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            d.rectangle([x0 + 2, y0 + 2, x1 - 2, y0 + (y1 - y0) * 0.65], fill=(35, 55, 95))  # 青黒いパネル
    arr = np.asarray(im)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            t = arr[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE]
            p = root / "tiles" / str(z) / str(tx) / f"{ty}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(t).save(p, quality=95)

    # --- 基盤地図情報 BldA (FG-GML 風。posList は 緯度 経度。外周を2セグメントに分割)
    ns = ('xmlns="http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema" '
          'xmlns:gml="http://www.opengis.net/gml/3.2"')
    out = [f'<?xml version="1.0" encoding="UTF-8"?><Dataset {ns}>']
    for i, (corners, k, panel, kind) in enumerate(blds):
        ring = corners + [corners[0]]
        seg = lambda pts: " ".join(f"{la:.7f} {lo:.7f}" for lo, la in pts)
        out.append(
            f'<BldA gml:id="b{i}"><fid>{i}</fid><type>普通建物</type><area><gml:Surface><gml:patches>'
            f'<gml:PolygonPatch><gml:exterior><gml:Ring><gml:curveMember><gml:Curve><gml:segments>'
            f'<gml:LineStringSegment><gml:posList>{seg(ring[:3])}</gml:posList></gml:LineStringSegment>'
            f'<gml:LineStringSegment><gml:posList>{seg(ring[2:])}</gml:posList></gml:LineStringSegment>'
            f'</gml:segments></gml:Curve></gml:curveMember></gml:Ring></gml:exterior>'
            f'</gml:PolygonPatch></gml:patches></gml:Surface></area></BldA>')
    out.append("</Dataset>")
    gml = root / "FG-GML-0000-00-BldA-test.xml"
    gml.write_text("".join(out), encoding="utf-8")

    # --- 一戸建世帯数 CSV（統計GIS 風: ID行 + 名称行）, cp932
    rows = [["KEY_CODE", "T001"], ["KEY_CODE", "一戸建"]]
    for key, k in zip(boundary.KEY_CODE, towns):
        rows.append([key, str(DETACHED[k])])
    pd.DataFrame(rows).to_csv(root / "stats.csv", header=False, index=False, encoding="cp932")
    n_house = {k: sum(1 for b in blds if b[1] == k and b[3] == "house") for k in towns}
    n_panel = {k: sum(1 for b in blds if b[1] == k and b[2]) for k in towns}
    return {"gml": gml, "csv": root / "stats.csv", "boundary": root / "boundary.gpkg",
            "n_house": n_house, "n_panel": n_panel, "n_total": len(blds)}
