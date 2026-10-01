"""e-Stat: 小地域境界(町丁目)と一戸建世帯数。"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from .config import CRS_GEO, ESTAT_BOUNDARY_URL, ESTAT_STATS_URL, USER_AGENT


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_boundary(code: str, out_dir: Path) -> Path:
    """小地域境界 shapefile (zip) を取得して展開し、.shp のパスを返す。"""
    zpath = _download(ESTAT_BOUNDARY_URL.format(code=code), out_dir / f"boundary_{code}.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir / f"boundary_{code}")
    shp = next((out_dir / f"boundary_{code}").rglob("*.shp"))
    return shp


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    """境界を読み込み、KEY_CODE / S_NAME を持つ JGD2011 緯度経度の GeoDataFrame にする。"""
    gdf = gpd.read_file(path, encoding="cp932") if str(path).endswith(".shp") else gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4612")
    gdf = gdf.to_crs(CRS_GEO)
    if "KEY_CODE" not in gdf:
        raise ValueError("境界データに KEY_CODE 列がありません")
    gdf["KEY_CODE"] = gdf["KEY_CODE"].astype(str)
    if "S_NAME" not in gdf:
        gdf["S_NAME"] = gdf["KEY_CODE"]
    # 水面調査区(HCODE=8154)は除外
    if "HCODE" in gdf:
        gdf = gdf[gdf["HCODE"].astype(str) != "8154"]
    # 同一 KEY_CODE が複数ポリゴンに分かれる場合は融合
    gdf = gdf.dissolve(by="KEY_CODE", as_index=False, aggfunc="first")
    return gdf[["KEY_CODE", "S_NAME", "geometry"]]


def download_stats(stats_id: str, code: str, out_dir: Path) -> Path:
    return _download(ESTAT_STATS_URL.format(stats_id=stats_id, code=code),
                     out_dir / f"stats_{stats_id}_{code}.csv")


def load_detached_households(path: Path, column_keyword: str = "一戸建") -> pd.DataFrame:
    """統計GIS の小地域 CSV から「一戸建」世帯数を取り出す。

    CSV は 1 行目に項目 ID(KEY_CODE, T001...)、2 行目に日本語項目名、3 行目以降がデータ
    という形式を想定する（2 行目が数値なら 1 行ヘッダとして扱う）。
    列名に column_keyword を含む列のうち、先頭のものを使う。複数ある場合は警告し候補を表示。
    """
    raw = pd.read_csv(path, header=None, dtype=str, encoding="cp932")
    ids = raw.iloc[0].fillna("").tolist()
    second = raw.iloc[1].fillna("").tolist()
    has_names = not any(v.replace(",", "").strip().isdigit() for v in second[1:5] if v)
    names = second if has_names else ids
    body = raw.iloc[2:] if has_names else raw.iloc[1:]
    if "KEY_CODE" not in ids:
        raise ValueError(f"KEY_CODE 列が見つかりません: {ids[:5]}")
    cands = [i for i, n in enumerate(names) if column_keyword in n]
    if not cands:
        raise ValueError(f"'{column_keyword}' を含む列がありません。列名: {names}")
    if len(cands) > 1:
        print("[estat] 候補が複数あります。先頭を使用:", [names[i] for i in cands])
    ci = cands[0]
    out = pd.DataFrame({
        "KEY_CODE": body.iloc[:, ids.index("KEY_CODE")].astype(str).str.strip(),
        "detached_households": pd.to_numeric(
            body.iloc[:, ci].str.replace(",", "", regex=False), errors="coerce"),
    })
    print(f"[estat] 一戸建列: {names[ci]!r}  合計={out['detached_households'].sum():.0f}")
    return out
