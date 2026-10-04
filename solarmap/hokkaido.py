"""北海道全域: 町丁目(小地域)ごとの一戸建世帯数を作る（2020 年国勢調査, e-Stat 統計GIS）。

e-Stat の小地域は「大字 > 丁目」の階層で秘匿(X)もあるため、
 * 境界ポリゴン(= 末端の区画)だけを区画として使い、
 * 市区町村の総数(正確)に合わせて、秘匿分を世帯数比で配分し全体をスケールする。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS_AREA, CRS_GEO, ESTAT_BOUNDARY_URL, ESTAT_STATS_URL
from .estat import _download

PREF = "01"


def _read_stats(txt: Path, col: int) -> pd.DataFrame:
    raw = pd.read_csv(txt, header=None, dtype=str, encoding="cp932")
    d = raw.iloc[2:, [0, 2, col]].copy()
    d.columns = ["KEY_CODE", "CITYNAME", "v"]
    d["v"] = pd.to_numeric(d["v"].str.replace(",", "", regex=False).replace("-", "0"), errors="coerce")
    d["KEY_CODE"] = d["KEY_CODE"].str.strip()
    return d


def build_units(work: Path) -> gpd.GeoDataFrame:
    """work/hokkaido/units.gpkg を作る。列: KEY_CODE, S_NAME, muni, city_name, detached(配分後)。"""
    raw = work / "raw"
    bz = _download(ESTAT_BOUNDARY_URL.format(code=PREF), raw / "boundary_01.zip")
    sz = _download(ESTAT_STATS_URL.format(stats_id="T001086", code=PREF), raw / "stats_T001086_01.zip")
    with zipfile.ZipFile(bz) as z:
        z.extractall(raw / "boundary_01")
    with zipfile.ZipFile(sz) as z:
        z.extractall(raw / "stats_T001086_01")
    g = gpd.read_file(next((raw / "boundary_01").rglob("*.shp")), encoding="cp932").to_crs(CRS_GEO)
    g["KEY_CODE"] = g["KEY_CODE"].astype(str)
    g = g[g["HCODE"].astype(str) != "8154"]            # 水面調査区
    g = g.dissolve(by="KEY_CODE", as_index=False, aggfunc="first")

    st = _read_stats(next((raw / "stats_T001086_01").glob("*.txt")), 8)   # 8 列目 = 一戸建
    muni = st[st["KEY_CODE"].str.len() == 5].set_index("KEY_CODE")
    leaf = st[st["KEY_CODE"].str.len() > 5].set_index("KEY_CODE")["v"]
    g["muni"] = g["KEY_CODE"].str[:5]
    g["detached_raw"] = g["KEY_CODE"].map(leaf)
    g["city_name"] = g["muni"].map(muni["CITYNAME"])
    g["total"] = g["muni"].map(muni["v"])

    # 秘匿(NaN)は SETAI で重み付けして配分 → 市区町村総数に合わせてスケール
    setai = pd.to_numeric(g["SETAI"], errors="coerce").fillna(0)
    out = np.zeros(len(g))
    for m, idx in g.groupby("muni").indices.items():
        sub = g.iloc[idx]
        total = sub["total"].iloc[0]
        if not np.isfinite(total):
            continue
        known = sub["detached_raw"].fillna(0).to_numpy()
        miss = sub["detached_raw"].isna().to_numpy()
        rem = max(total - known.sum(), 0.0)
        w = setai.iloc[idx].to_numpy() * miss
        est = known + (w / w.sum() * rem if w.sum() > 0 else 0)
        s = est.sum()
        out[idx] = est * (total / s) if s > 0 else 0
    g["detached"] = out
    g = g[["KEY_CODE", "S_NAME", "muni", "city_name", "detached_raw", "detached", "geometry"]]
    (work).mkdir(parents=True, exist_ok=True)
    g.to_file(work / "units.gpkg", driver="GPKG")
    muni.reset_index().rename(columns={"v": "detached"}).to_csv(work / "muni_detached.csv", index=False)
    return g
