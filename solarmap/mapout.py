"""GeoJSON と folium HTML の出力。"""
from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.element import MacroElement, Template

from .aggregate import BINS, COLORS, NA_COLOR

_FIELDS = ["S_NAME", "n_panel", "n_candidates", "detached_households", "rate_pct"]
_ALIASES = ["町丁目", "パネルあり(件)", "判定した戸建て候補(件)", "一戸建世帯数", "設置率(%)"]


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_file(path, driver="GeoJSON")


def _legend() -> MacroElement:
    labels = ["0〜2%", "2〜5%", "5〜8%", "8〜12%", "12〜20%", "20%〜"]
    rows = "".join(
        f'<div><i style="background:{c};width:14px;height:14px;display:inline-block;'
        f'margin-right:6px;border:1px solid #888"></i>{l}</div>' for c, l in zip(COLORS, labels))
    rows += (f'<div><i style="background:{NA_COLOR};width:14px;height:14px;display:inline-block;'
             f'margin-right:6px;border:1px solid #888"></i>算出不可</div>')
    t = Template(f"""{{% macro html(this, kwargs) %}}
    <div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#fff;
      padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px sans-serif">
      <b>太陽光パネル設置率</b><br>(パネルあり / 一戸建世帯数)<br>{rows}</div>
    {{% endmacro %}}""")
    m = MacroElement()
    m._template = t
    return m


def write_html(gdf: gpd.GeoDataFrame, path: Path, panel_buildings: gpd.GeoDataFrame | None = None,
               title: str = "札幌市 住宅用太陽光パネル設置率（パイロット）") -> None:
    g = gdf.copy()
    g["rate_pct"] = (g["rate"] * 100).round(1)
    for c in ("rate_pct", "detached_households"):
        g[c] = g[c].astype(object).where(g[c].notna(), "-")
    minx, miny, maxx, maxy = gdf.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=13,
                   tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
        attr='<a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>',
        name="空中写真(国土地理院)", max_zoom=18, show=False).add_to(m)
    folium.GeoJson(
        g[["S_NAME", "n_panel", "n_candidates", "detached_households", "rate_pct", "fill", "geometry"]],
        name="町丁目別設置率",
        style_function=lambda f: {"fillColor": f["properties"]["fill"], "color": "#555",
                                  "weight": 1, "fillOpacity": 0.7},
        highlight_function=lambda f: {"weight": 3, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(fields=_FIELDS, aliases=_ALIASES, localize=True),
    ).add_to(m)
    if panel_buildings is not None and len(panel_buildings):
        fg = folium.FeatureGroup(name="パネルあり判定の建物", show=False)
        for r in panel_buildings.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=2, color="#d62728", weight=1,
                                tooltip=f"{r.bid} score={r.score:.2f}").add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    m.get_root().add_child(_legend())
    m.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
    m.save(str(path))
