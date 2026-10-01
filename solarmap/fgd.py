"""基盤地図情報（建物の外周線 BldA）GML の読み込み。"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import geopandas as gpd
from shapely.geometry import Polygon

from .config import CRS_GEO


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ring(elem: ET.Element) -> list[tuple[float, float]]:
    """exterior/interior 配下の posList を連結して (lon, lat) の列にする。"""
    coords: list[tuple[float, float]] = []
    for pl in elem.iter():
        if _local(pl.tag) != "posList" or not pl.text:
            continue
        v = [float(t) for t in pl.text.split()]
        pts = [(v[i + 1], v[i]) for i in range(0, len(v) - 1, 2)]  # GML は 緯度 経度 の順
        if coords and pts and coords[-1] == pts[0]:
            pts = pts[1:]
        coords.extend(pts)
    return coords


def _polygons(bld: ET.Element) -> Iterator[Polygon]:
    # Surface/PolygonPatch (基盤地図情報 v4) と Polygon の両方を exterior/interior で処理
    holders = [e for e in bld.iter() if _local(e.tag) in ("PolygonPatch", "Polygon")]
    for h in holders:
        ext, holes = None, []
        for ch in h:
            if _local(ch.tag) == "exterior":
                ext = _ring(ch)
            elif _local(ch.tag) == "interior":
                holes.append(_ring(ch))
        if ext and len(ext) >= 4:
            try:
                yield Polygon(ext, [h_ for h_ in holes if len(h_) >= 4])
            except ValueError:
                continue


def _parse_xml(stream) -> Iterator[dict]:
    for _, el in ET.iterparse(stream, events=("end",)):
        if _local(el.tag) != "BldA":
            continue
        btype = next((c.text for c in el if _local(c.tag) == "type"), None)
        fid = next((c.text for c in el if _local(c.tag) == "fid"), None)
        for poly in _polygons(el):
            yield {"fgd_fid": fid, "bld_type": btype, "geometry": poly}
        el.clear()


def _iter_streams(paths: Iterable[Path]):
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.lower().endswith(".xml") and "BldA" in n:
                        with z.open(n) as f:
                            yield n, f
        elif p.suffix.lower() == ".xml":
            with open(p, "rb") as f:
                yield p.name, f


def load_bldA(paths: Iterable[Path]) -> gpd.GeoDataFrame:
    """FG-GML の BldA (.xml / .zip) を GeoDataFrame(EPSG:6668) にする。"""
    rows = []
    for name, f in _iter_streams(paths):
        n0 = len(rows)
        rows.extend(_parse_xml(f))
        print(f"[fgd] {name}: {len(rows) - n0} 件")
    if not rows:
        raise ValueError("BldA が 1 件も読めませんでした（ファイルに BldA を含めてください）")
    return gpd.GeoDataFrame(rows, crs=CRS_GEO)
