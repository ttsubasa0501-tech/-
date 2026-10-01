"""検出精度の目視確認（100 件）用 HTML と集計。

層化抽出: 「パネルあり判定」と「なし判定」から半数ずつ抽出する。
 - あり判定層 → 適合率(precision)
 - なし判定層 → 見逃し率 → 全体の再現率(recall)の推定（層の母数で重み付け）
全体からのランダム 100 件だと、設置率が数%の地域ではパネルありが数件しか入らず精度が評価できないため。
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .tiles import TileStore, building_crop

SCALE = 4  # 目視用の拡大率


def sample(pred: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    p = pred[pred["valid"]]
    pos, neg = p[p["pred"]], p[~p["pred"]]
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n - n_pos)
    n_pos = min(len(pos), n - n_neg)  # 片側が足りなければ他方で補う
    s = pd.concat([pos.sample(n_pos, random_state=seed), neg.sample(n_neg, random_state=seed)])
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)


def _card_image(row, geom, store: TileStore, out: Path) -> None:
    c = building_crop(geom, store, margin=16)
    raw = Image.fromarray(c.img).resize((c.img.shape[1] * SCALE, c.img.shape[0] * SCALE), Image.LANCZOS)
    ann = raw.copy()
    ImageDraw.Draw(ann).line([(x * SCALE, y * SCALE) for x, y in c.poly_px + [c.poly_px[0]]],
                             fill=(255, 230, 0), width=2)
    w, h = raw.size
    canvas = Image.new("RGB", (w * 2 + 6, h), (255, 255, 255))
    canvas.paste(raw, (0, 0))
    canvas.paste(ann, (w + 6, 0))
    canvas.save(out, quality=88)


def build_review(pred: pd.DataFrame, bld: gpd.GeoDataFrame, store: TileStore, out_dir: Path,
                 n: int = 100, seed: int = 0, detector_name: str = "", threshold: float = 0.5) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    s = sample(pred, n, seed)
    b = bld.set_index("bid")
    valid = pred[pred["valid"]]
    totals = {"pos": int(valid["pred"].sum()), "neg": int((~valid["pred"]).sum())}

    items = []
    for r in s.itertuples():
        g = b.loc[r.bid]
        _card_image(r, g.geometry, store, out_dir / "img" / f"{r.bid}.jpg")
        items.append({"bid": r.bid, "pred": bool(r.pred), "score": round(float(r.score), 3),
                      "lat": round(float(g.lat), 6), "lon": round(float(g.lon), 6),
                      "area": round(float(g.area_m2), 1), "town": g.get("S_NAME", "")})
    pd.DataFrame(items).to_csv(out_dir / "review_sample.csv", index=False)
    page = out_dir / "review.html"
    page.write_text(_HTML.replace("__DATA__", json.dumps({"items": items, "totals": totals},
                                                         ensure_ascii=False))
                    .replace("__META__", html.escape(f"{detector_name} / threshold={threshold}")),
                    encoding="utf-8")
    return page


def evaluate(labels: pd.DataFrame, totals: dict) -> dict:
    """labels: bid, pred(bool), truth ('yes'/'no')。'unsure' は除外。"""
    d = labels[labels["truth"].isin(["yes", "no"])]
    pos, neg = d[d["pred"]], d[~d["pred"]]
    tp, fp = int((pos["truth"] == "yes").sum()), int((pos["truth"] == "no").sum())
    fn, tn = int((neg["truth"] == "yes").sum()), int((neg["truth"] == "no").sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    miss = fn / (fn + tn) if fn + tn else float("nan")
    est_tp, est_fn = totals["pos"] * prec, totals["neg"] * miss
    rec = est_tp / (est_tp + est_fn) if est_tp + est_fn else float("nan")
    return {"precision": prec, "miss_rate_in_negatives": miss, "est_recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unsure": int(len(labels) - len(d))}


_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>パネル判定 目視確認</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:14px/1.5 sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 16px;z-index:5}
#sum{font-size:13px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;padding:12px}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px}
.card img{width:100%;image-rendering:auto}
.pos{border-left:6px solid #d62728}.neg{border-left:6px solid #1f77b4}
.meta{font-size:12px;color:#444} label{margin-right:12px;cursor:pointer}
button{padding:4px 10px}
</style></head><body>
<header><b>太陽光パネル判定の目視確認</b> <span class="meta">(__META__)</span><br>
左: 元画像 / 右: 建物外周(黄)。各建物に「実際にパネルがあるか」を選択してください。
<div id="sum"></div>
<button onclick="dl()">ラベルをCSV保存</button></header>
<div class="grid" id="g"></div>
<script>
const D=__DATA__, KEY="solar_review_v1";
let L=JSON.parse(localStorage.getItem(KEY)||"{}");
const g=document.getElementById("g");
D.items.forEach((it,i)=>{
 const c=document.createElement("div");c.className="card "+(it.pred?"pos":"neg");
 const u=`https://maps.gsi.go.jp/#19/${it.lat}/${it.lon}/&base=seamlessphoto`;
 c.innerHTML=`<img loading="lazy" src="img/${it.bid}.jpg">
 <div class="meta">#${i+1} ${it.bid} ${it.town||""} / 面積${it.area}㎡ / モデル判定: <b>${it.pred?"あり":"なし"}</b> (score ${it.score}) / <a href="${u}" target="_blank">地理院地図</a></div>
 <div>${[["yes","パネルあり"],["no","パネルなし"],["unsure","判別不能"]].map(([v,t])=>
 `<label><input type="radio" name="${it.bid}" value="${v}" ${L[it.bid]==v?"checked":""}>${t}</label>`).join("")}</div>`;
 c.querySelectorAll("input").forEach(r=>r.onchange=()=>{L[it.bid]=r.value;localStorage.setItem(KEY,JSON.stringify(L));sum();});
 g.appendChild(c);});
function f(x){return isFinite(x)?(x*100).toFixed(1)+"%":"-"}
function sum(){
 let tp=0,fp=0,fn=0,tn=0,un=0,done=0;
 D.items.forEach(it=>{const t=L[it.bid];if(!t)return;done++;if(t=="unsure"){un++;return}
  if(it.pred){t=="yes"?tp++:fp++}else{t=="yes"?fn++:tn++}});
 const prec=tp/(tp+fp),miss=fn/(fn+tn);
 const etp=D.totals.pos*prec,efn=D.totals.neg*miss,rec=etp/(etp+efn);
 document.getElementById("sum").innerHTML=`確認 ${done}/${D.items.length}（判別不能${un}） | 適合率(あり判定の正解率) <b>${f(prec)}</b> (TP${tp}/FP${fp}) | なし判定の見逃し率 <b>${f(miss)}</b> (FN${fn}/TN${tn}) | 推定再現率 <b>${f(rec)}</b>`;}
function dl(){let s="bid,pred,truth\n"+D.items.filter(i=>L[i.bid]).map(i=>`${i.bid},${i.pred},${L[i.bid]}`).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download="review_labels.csv";a.click();}
sum();
</script></body></html>"""
