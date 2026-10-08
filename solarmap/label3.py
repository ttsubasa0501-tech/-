"""築年・屋根の形・太陽光を 1 軒ずつまとめて目視ラベルするページ（手稲区）。

層(重複なし): ai=AIスコア0.3以上 / age=築年の予備スコア上位(ai以外) / rest=その他。
層ごとに抽出数と母数を保存しておき、学習後の精度を重み付けして推定する。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .age import OLD_URL, _pair_image, prelim_score
from .tiles import TileStore


def sample(feat: pd.DataFrame, pred: pd.DataFrame, n_rest: int = 70, n_age: int = 65, n_ai: int = 65,
           ai_thr: float = 0.3, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    d = feat.merge(pred[["bid", "score", "valid"]], on="bid").dropna(subset=["cont_old"])
    d = d[d["valid"]].copy()
    d["prelim"] = prelim_score(d)
    d["stratum"] = "rest"
    d.loc[d["prelim"] >= d["prelim"].quantile(0.75), "stratum"] = "age"
    d.loc[d["score"] >= ai_thr, "stratum"] = "ai"
    pop = d["stratum"].value_counts().to_dict()
    take = {"ai": n_ai, "age": n_age, "rest": n_rest}
    s = pd.concat([g.sample(min(take[k], len(g)), random_state=seed) for k, g in d.groupby("stratum")])
    return s.sample(frac=1, random_state=seed).reset_index(drop=True), {"population": pop, "sampled": take, "ai_thr": ai_thr}


def build_page(samples: pd.DataFrame, meta: dict, buildings: gpd.GeoDataFrame, work: Path, year: int, out: Path) -> Path:
    work = Path(work)
    new = TileStore(work / "tiles", offline=True)
    old = TileStore(work / f"tiles_{year}", url=OLD_URL.replace("{year}", str(year)), offline=True, ext="png")
    b = buildings.set_index("bid")
    items = []
    for r in samples.itertuples():
        g = b.loc[r.bid]
        items.append({"bid": r.bid, "town": g.get("S_NAME", ""), "area": round(float(g.area_m2)),
                      "lat": round(float(g.lat), 6), "lon": round(float(g.lon), 6), "stratum": r.stratum,
                      "src": "data:image/jpeg;base64," + base64.b64encode(_pair_image(g.geometry, old, new)).decode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in i.items() if k != "src"} for i in items]).to_csv(out.with_name("label3_sample.csv"), index=False)
    (out.with_name("label3_meta.json")).write_text(json.dumps(meta, ensure_ascii=False))
    out.write_text(_PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace("__YEAR__", str(year)), encoding="utf-8")
    return out


_PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>まとめて目視ラベル（築年・屋根・太陽光）</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px/1.5 sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 16px;z-index:5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:12px;padding:12px}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px}.card.done{border-color:#2ca02c;background:#f6fff6}.card img{width:100%}
.meta{font-size:12px;color:#444}.q{margin:3px 0;padding:2px 0;border-top:1px dashed #ddd}.q b{display:inline-block;min-width:112px}
label{margin-right:8px;cursor:pointer;white-space:nowrap}button{padding:4px 10px}</style></head><body>
<header><b>まとめて目視ラベル（1軒につき3問）</b><br>
<span class="meta">画像: 左=<b>__YEAR__年度</b>の写真 / 右=<b>現在</b>の写真。黄枠は同じ建物の位置。<br>
<b>①築年（左の写真）</b> あり=同じ場所に建物 / 建て替え=別の形の建物 / なし=空き地・畑・駐車場・森 / 不明<br>
<b>②屋根の形（右の写真）</b> 陸屋根(無落雪を含む。棟や傾斜面が見えない箱形) / 勾配(切妻・寄棟など) / 不明<br>
<b>③太陽光パネル（右の写真）</b> あり=屋根に黒/濃紺の格子状のパネルが見える / なし / 不明 ※温室・物置など住宅でない建物は全て「不明」で構いません</span>
<div id="sum"></div><button onclick="dl()">ラベルをCSV保存</button></header>
<div class="grid" id="g"></div>
<script>
const D=__DATA__,KEY="label3___YEAR__";let L=JSON.parse(localStorage.getItem(KEY)||"{}");const g=document.getElementById("g");
const Q=[["age","①築年",[["present","あり"],["rebuilt","建て替え"],["absent","なし"],["unsure","不明"]]],
 ["roof","②屋根",[["flat","陸屋根"],["pitched","勾配"],["unsure","不明"]]],
 ["solar","③太陽光",[["yes","あり"],["no","なし"],["unsure","不明"]]]];
D.forEach((it,i)=>{const c=document.createElement("div");c.className="card";c.id="c_"+it.bid;
 const u=`https://maps.gsi.go.jp/#19/${it.lat}/${it.lon}/&base=seamlessphoto`;
 c.innerHTML=`<img loading="lazy" src="${it.src}"><div class="meta">#${i+1} ${it.town} / ${it.area}㎡ / <a href="${u}" target="_blank">地理院地図</a></div>`+
 Q.map(([k,t,o])=>`<div class="q"><b>${t}</b>${o.map(([v,l])=>`<label><input type="radio" name="${it.bid}_${k}" value="${v}" ${(L[it.bid]||{})[k]==v?"checked":""}>${l}</label>`).join("")}</div>`).join("");
 c.querySelectorAll("input").forEach(r=>r.onchange=()=>{const[b,k]=r.name.split("_");L[b]=L[b]||{};L[b][k]=r.value;localStorage.setItem(KEY,JSON.stringify(L));sum()});
 g.appendChild(c)});
function sum(){let all=0,part=0;D.forEach(it=>{const l=L[it.bid]||{};const n=["age","roof","solar"].filter(k=>l[k]).length;if(n===3){all++;document.getElementById("c_"+it.bid).classList.add("done")}else{document.getElementById("c_"+it.bid).classList.remove("done");if(n)part++}});
 document.getElementById("sum").innerHTML=`3問とも回答済み ${all}/${D.length}（一部のみ ${part}）`}
function dl(){let s="bid,stratum,age,roof,solar\n"+D.filter(i=>L[i.bid]).map(i=>{const l=L[i.bid];return[i.bid,i.stratum,l.age||"",l.roof||"",l.solar||""].join(",")}).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download="label3.csv";a.click()}
sum();</script></body></html>"""
