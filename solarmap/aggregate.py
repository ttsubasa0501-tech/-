"""町丁目ごとの集計と色付け。"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

# 設置率の階級(上端)と色(明→暗)。NaN(分母なし)は灰色。
BINS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 1.0e9]
COLORS = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
NA_COLOR = "#cccccc"


def rate_color(rate: float) -> str:
    if rate is None or not np.isfinite(rate):
        return NA_COLOR
    for hi, c in zip(BINS[1:], COLORS):
        if rate < hi:
            return c
    return COLORS[-1]


def aggregate(boundary: gpd.GeoDataFrame, bld: gpd.GeoDataFrame, pred: pd.DataFrame,
              households: pd.DataFrame) -> gpd.GeoDataFrame:
    """設置率 = パネルあり件数 / 一戸建世帯数。

    pred: bid, score, pred(bool), valid(bool)。valid=False(タイル欠損)は集計から除外。
    """
    b = bld[["bid", "KEY_CODE"]].merge(pred, on="bid", how="left")
    b = b[b["valid"].fillna(False)]
    g = b.groupby("KEY_CODE").agg(n_candidates=("bid", "size"), n_panel=("pred", "sum")).reset_index()
    out = boundary.merge(g, on="KEY_CODE", how="left").merge(households, on="KEY_CODE", how="left")
    for c in ("n_candidates", "n_panel"):
        out[c] = out[c].fillna(0).astype(int)
    d = out["detached_households"]
    out["rate"] = np.where(d > 0, out["n_panel"] / d.where(d > 0), np.nan)
    out["fill"] = out["rate"].map(rate_color)
    out["fill-opacity"] = 0.7
    out["stroke"] = "#555555"
    return out
