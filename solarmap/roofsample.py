"""屋根形状(陸屋根/勾配屋根)の割合を測るための標本抽出と目視ラベルページ。

北海道全域の戸建てに対して代表的になるよう、一戸建世帯数に比例して区画を選び、
その区画内の戸建てサイズの建物を 1 棟ずつ抽出する。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .buildings import extract_detached_candidates
from .config import CRS_AREA, CRS_GEO
from .tiles import TileStore, building_crop
from .vtile import ZOOM, decode_tile, fetch_tile, tile_of

SCALE = 4


def sample_houses(units: gpd.GeoDataFrame, cache: Path, n: int = 150, seed: int = 0) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    u = units[units["detached"] > 0].reset_index(drop=True)
    order = rng.choice(len(u), size=len(u), replace=False, p=u["detached"] / u["detached"].sum())
    out = []
    for i in order:
        if len(out) >= n:
            break
        unit = u.iloc[i]
        p = unit.geometry.representative_point()
        tx, ty = tile_of(p.x, p.y)
        rows, _ = decode_tile(fetch_tile(tx, ty, cache), tx, ty)
        if not rows:
            continue
        b = gpd.GeoDataFrame(rows, crs=CRS_GEO)
        b = b[b["bld_type"].isin(["普通建物", "堅ろう建物"])]
        if b.empty:
            continue
        bounds = gpd.GeoDataFrame({"KEY_CODE": [unit.KEY_CODE]}, geometry=[unit.geometry], crs=CRS_GEO)
        cand = extract_detached_candidates(b, bounds)   # 区画内・戸建てサイズ・コンパクト形状
        if cand.empty:
            continue
        pick = cand.iloc[int(rng.integers(len(cand)))]
        out.append({"bid": f"r{len(out):03d}", "KEY_CODE": unit.KEY_CODE, "S_NAME": unit.S_NAME,
                    "muni": unit.muni, "city_name": unit.city_name, "area_m2": pick.area_m2,
                    "lon": pick.lon, "lat": pick.lat, "geometry": pick.geometry})
    return gpd.GeoDataFrame(out, crs=CRS_GEO)


def _image(geom, store: TileStore) -> bytes:
    c = building_crop(geom, store, margin=18)
    raw = Image.fromarray(c.img).resize((c.img.shape[1] * SCALE, c.img.shape[0] * SCALE), Image.LANCZOS)
    ann = raw.copy()
    ImageDraw.Draw(ann).line([(x * SCALE, y * SCALE) for x, y in c.poly_px + [c.poly_px[0]]],
                             fill=(255, 230, 0), width=2)
    w, h = raw.size
    canvas = Image.new("RGB", (w * 2 + 6, h), "white")
    canvas.paste(raw, (0, 0))
    canvas.paste(ann, (w + 6, 0))
    from io import BytesIO
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def build_page(samples: gpd.GeoDataFrame, store: TileStore, out: Path) -> Path:
    items = []
    for r in samples.itertuples():
        items.append({"bid": r.bid, "city": r.city_name, "town": r.S_NAME, "muni": r.muni,
                      "sapporo": r.muni.startswith("011") and 1101 <= int(r.muni[1:]) <= 1110,
                      "area": round(r.area_m2), "lat": round(r.lat, 6), "lon": round(r.lon, 6),
                      "src": "data:image/jpeg;base64," + base64.b64encode(_image(r.geometry, store)).decode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in i.items() if k != "src"} for i in items]).to_csv(
        out.with_name("roof_sample.csv"), index=False)
    out.write_text(_HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
    return out


_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>屋根形状の目視ラベル</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px/1.5 sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 16px;z-index:5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;padding:12px}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px}.card img{width:100%}
.meta{font-size:12px;color:#444}label{margin-right:12px;cursor:pointer}button{padding:4px 10px}</style></head><body>
<header><b>屋根形状の目視ラベル</b><br>
<span class="meta">左: 元画像 / 右: 対象建物(黄枠)。黄枠の建物の屋根が<b>陸屋根</b>(平ら・無落雪屋根を含む。屋根の勾配や棟が見えない箱形)か
<b>勾配屋根</b>(切妻・寄棟など棟や傾斜面が見える)かを選択。屋根は見えるが形が判断できなければ「判別不能」、温室・物置・駐車場など住宅でない/写真に建物が無い場合は「住宅でない/建物なし」(集計から除外)。</span>
<div id="sum"></div><button onclick="dl()">ラベルをCSV保存</button></header>
<div class="grid" id="g"></div>
<script>
const D=__DATA__,KEY="roof_label_v1";let L=JSON.parse(localStorage.getItem(KEY)||"{}");
const g=document.getElementById("g");
D.forEach((it,i)=>{const c=document.createElement("div");c.className="card";
 const u=`https://maps.gsi.go.jp/#19/${it.lat}/${it.lon}/&base=seamlessphoto`;
 c.innerHTML=`<img loading="lazy" src="${it.src}"><div class="meta">#${i+1} ${it.city} ${it.town} / ${it.area}㎡ / <a href="${u}" target="_blank">地理院地図</a></div>
 <div>${[["flat","陸屋根"],["pitched","勾配屋根"],["unsure","判別不能"],["notHouse","住宅でない/建物なし"]].map(([v,t])=>`<label><input type="radio" name="${it.bid}" value="${v}" ${L[it.bid]==v?"checked":""}>${t}</label>`).join("")}</div>`;
 c.querySelectorAll("input").forEach(r=>r.onchange=()=>{L[it.bid]=r.value;localStorage.setItem(KEY,JSON.stringify(L));sum();});g.appendChild(c);});
function wil(k,n){if(!n)return"-";const z=1.96,p=k/n,d=1+z*z/n,c=p+z*z/(2*n),a=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));
 return (100*p).toFixed(0)+"% ("+(100*(c-a)/d).toFixed(0)+"〜"+(100*(c+a)/d).toFixed(0)+"%)";}
function sum(){const grp=(f)=>{let k=0,n=0;D.filter(f).forEach(it=>{const t=L[it.bid];if(t=="flat"){k++;n++}else if(t=="pitched")n++});return[k,n]};
 const [a,an]=grp(()=>true),[s,sn]=grp(i=>i.sapporo),[o,on]=grp(i=>!i.sapporo);
 const done=D.filter(i=>L[i.bid]).length;
 document.getElementById("sum").innerHTML=`確認 ${done}/${D.length} | 陸屋根率(判別可のみ) 全体 <b>${wil(a,an)}</b> n=${an} | 札幌市 <b>${wil(s,sn)}</b> n=${sn} | 札幌市以外 <b>${wil(o,on)}</b> n=${on}`;}
function dl(){let s="bid,muni,city,sapporo,label\n"+D.filter(i=>L[i.bid]).map(i=>`${i.bid},${i.muni},${i.city},${i.sapporo},${L[i.bid]}`).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download="roof_labels.csv";a.click();}
sum();</script></body></html>"""
