"""建物の戸建てサイズ抽出と町丁目への割り当て。"""
from __future__ import annotations

import geopandas as gpd

from .config import CRS_AREA, CRS_GEO


def extract_detached_candidates(
    bld: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    min_area: float = 50.0,
    max_area: float = 250.0,
    min_compactness: float = 0.6,
    types: tuple[str, ...] | None = None,
) -> gpd.GeoDataFrame:
    """戸建てらしい建物を抽出し KEY_CODE を付与する。

    * 面積は建築面積(外周線の面積)で代用する。床面積は 2 階建てなら約 2 倍になる点に注意。
    * min_compactness: 面積 / 最小回転外接矩形の面積。細長い/複雑な形(倉庫・連棟等)を除く。
    * types: BldA の type で絞る場合に指定（例: ("普通建物",)）。
    """
    b = bld.to_crs(CRS_AREA)
    b["area_m2"] = b.geometry.area
    b = b[(b["area_m2"] >= min_area) & (b["area_m2"] <= max_area)]
    if types:
        b = b[b["bld_type"].isin(types)]
    mrr = b.geometry.minimum_rotated_rectangle().area
    b = b[(b["area_m2"] / mrr) >= min_compactness].copy()

    # 重心が入る町丁目に割り当て（範囲外は除外）
    cent = gpd.GeoDataFrame(geometry=b.geometry.centroid, index=b.index, crs=CRS_AREA)
    keys = boundary.to_crs(CRS_AREA)[["KEY_CODE", "geometry"]]
    joined = gpd.sjoin(cent, keys, how="inner", predicate="within")
    joined = joined[~joined.index.duplicated()]
    b = b.loc[joined.index].copy()
    b["KEY_CODE"] = joined["KEY_CODE"]

    b = b.reset_index(drop=True)
    c = b.geometry.centroid.to_crs(CRS_GEO)   # 重心は投影座標系で計算してから経緯度へ
    b = b.to_crs(CRS_GEO)
    b.insert(0, "bid", [f"b{i:06d}" for i in range(len(b))])
    b["lon"], b["lat"] = c.x.values, c.y.values
    return b
