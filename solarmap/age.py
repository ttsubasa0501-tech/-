"""築年(築20年以内か)の家ごとの推定: 2008 年度の航空写真(国土地理院 年度別空中写真)に建物があったかを判定する。

2008 年に建物が無い(空き地・畑・工事前)なら、2008 年以降に建った = 築18年未満 = 「築20年以内」。
2008 年に建物があれば築18年以上。2006〜2008 年築(築18〜20年)は 2008 年に写っているため
「築20年超」側に入ってしまい、手稲区では全体の数%の取りこぼしになる（既知の限界）。
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from .tiles import TileStore, building_crop

OLD_URL = "https://cyberjapandata.gsi.go.jp/xyz/nendophoto{year}/{z}/{x}/{y}.png"
FEATURES = ["cont_old", "cont_old25", "cont_new", "cont_new25", "exg_old", "sat_old", "lum_old", "tex_old", "exg_new", "tex_new",
            "ncc", "dcol", "dlum", "std_diff", "grad_old", "grad_new"]

_G = {}


def _init(work: str, year: int):
    _G["new"] = TileStore(Path(work) / "tiles", offline=True)
    _G["old"] = TileStore(Path(work) / f"tiles_{year}", url=OLD_URL.replace("{year}", str(year)),
                          offline=True, ext="png")


def _stats(a: np.ndarray):
    r, g, b = a[:, 0].astype(float), a[:, 1].astype(float), a[:, 2].astype(float)
    tot = np.maximum(r + g + b, 1.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = a.max(axis=1).astype(float), a.min(axis=1).astype(float)
    return lum, float(np.mean((2 * g - r - b) / tot)), float(np.mean((mx - mn) / np.maximum(mx, 1)))


def _grad(img: np.ndarray, m: np.ndarray) -> float:
    lum = img.astype(float) @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(lum)
    return float(np.hypot(gx, gy)[m].mean())


def _contrast(img: np.ndarray, poly: list, mask: np.ndarray, step: float = 3.0) -> tuple[float, float]:
    """輪郭に沿って「内側 step px」と「外側 step px」の色差を測り、(中央値, 25パーセンタイル) を返す。

    建物があれば輪郭で屋根と地面の色が変わり色差が大きい。更地・畑なら内外で差が小さい。
    """
    pts = np.asarray(poly, float)
    h, w = img.shape[:2]
    diffs = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 2:
            continue
        t = (b - a) / L
        nrm = np.array([-t[1], t[0]])
        for u in np.arange(1.0, L, 2.0):
            c = a + t * u
            p1, p2 = c + nrm * step, c - nrm * step
            x1, y1, x2, y2 = int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1]))
            if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                continue
            i1, i2 = (p1, p2) if mask[y1, x1] else (p2, p1)   # 内側/外側
            ix, iy, ox, oy = int(round(i1[0])), int(round(i1[1])), int(round(i2[0])), int(round(i2[1]))
            if mask[oy, ox] or not mask[iy, ix]:
                continue
            diffs.append(float(np.abs(img[iy, ix].astype(float) - img[oy, ox].astype(float)).mean()))
    if len(diffs) < 4:
        return float("nan"), float("nan")
    d = np.asarray(diffs)
    return float(np.median(d)), float(np.percentile(d, 25))


def _one(args):
    bid, geom = args
    try:
        co = building_crop(geom, _G["old"], margin=8)
        cn = building_crop(geom, _G["new"], margin=8)
    except Exception:
        return {"bid": bid}
    if co.missing or cn.missing:
        return {"bid": bid}
    m = Image.fromarray((cn.mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))
    me = np.asarray(m) > 0
    mask = me if me.sum() >= 15 else cn.mask
    if mask.sum() < 8 or co.img.shape != cn.img.shape:
        return {"bid": bid}
    ao, an = co.img[mask], cn.img[mask]
    lo, exo, so = _stats(ao)
    ln, exn, _ = _stats(an)
    lum_o = co.img.astype(float) @ np.array([0.299, 0.587, 0.114])
    lum_n = cn.img.astype(float) @ np.array([0.299, 0.587, 0.114])
    best = -1.0
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            sh = np.roll(np.roll(lum_n, dy, 0), dx, 1)[mask]
            a, b = lum_o[mask], sh
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            best = max(best, float(np.corrcoef(a, b)[0, 1]))
    co_med, co_p25 = _contrast(co.img, co.poly_px, cn.mask)
    cn_med, cn_p25 = _contrast(cn.img, cn.poly_px, cn.mask)
    return {"bid": bid, "cont_old": co_med, "cont_old25": co_p25, "cont_new": cn_med, "cont_new25": cn_p25,
            "exg_old": exo, "sat_old": so, "lum_old": float(lo.mean()), "tex_old": float(lo.std()),
            "exg_new": exn, "tex_new": float(ln.std()), "ncc": best,
            "dcol": float(np.abs(ao.mean(0) - an.mean(0)).mean()), "dlum": float(lo.mean() - ln.mean()),
            "std_diff": float(abs(lo.std() - ln.std())),
            "grad_old": _grad(co.img, mask), "grad_new": _grad(cn.img, mask)}


def extract_features(buildings: gpd.GeoDataFrame, work: Path, year: int = 2008, workers: int = 4) -> pd.DataFrame:
    work = Path(work)
    with Pool(workers, initializer=_init, initargs=(str(work), year)) as pool:
        rows = pool.map(_one, list(zip(buildings["bid"], buildings.geometry)), chunksize=200)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 目視確認ページ
import base64
import json
from io import BytesIO

from PIL import ImageDraw


def prelim_score(f: pd.DataFrame) -> pd.Series:
    """ラベル無しの予備スコア(大きいほど「2008年に建物が無かった」らしい): 輪郭の色差が小さい・現在との比が小さい・平坦。"""
    ratio = f["cont_old"] / f["cont_new"].clip(lower=1)
    r = lambda s, asc: s.rank(ascending=asc, pct=True)
    return (r(f["cont_old"], False) + r(ratio, False) + r(f["tex_old"], False) + r(f["grad_old"], False)) / 4


def sample_for_review(f: pd.DataFrame, n: int = 120, top_share: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """予備スコア上位から top_share、残りから (1-top_share) をランダム抽出（新築が少ないので上位を厚めに）。"""
    f = f.dropna(subset=["cont_old"]).copy()
    f["prelim"] = prelim_score(f)
    cut = f["prelim"].quantile(0.75)
    top, rest = f[f["prelim"] >= cut], f[f["prelim"] < cut]
    nt = int(n * top_share)
    s = pd.concat([top.sample(nt, random_state=seed).assign(stratum="top"),
                   rest.sample(n - nt, random_state=seed).assign(stratum="rest")])
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)


def _pair_image(geom, old: TileStore, new: TileStore, scale: int = 4) -> bytes:
    outs = []
    for st in (old, new):
        c = building_crop(geom, st, margin=16)
        im = Image.fromarray(c.img).resize((c.img.shape[1] * scale, c.img.shape[0] * scale), Image.LANCZOS)
        ImageDraw.Draw(im).line([(x * scale, y * scale) for x, y in c.poly_px + [c.poly_px[0]]],
                                fill=(255, 230, 0), width=2)
        outs.append(im)
    w, h = outs[0].size
    cv = Image.new("RGB", (w * 2 + 6, h), "white")
    cv.paste(outs[0], (0, 0))
    cv.paste(outs[1], (w + 6, 0))
    buf = BytesIO()
    cv.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def build_review_page(samples: pd.DataFrame, buildings: gpd.GeoDataFrame, work: Path, year: int, out: Path) -> Path:
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
    pd.DataFrame([{k: v for k, v in i.items() if k != "src"} for i in items]).to_csv(
        out.with_name("age_sample.csv"), index=False)
    out.write_text(_PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace("__YEAR__", str(year)),
                   encoding="utf-8")
    return out


_PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>築年(__YEAR__年の写真)の目視ラベル</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px/1.5 sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 16px;z-index:5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:12px;padding:12px}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px}.card img{width:100%}
.meta{font-size:12px;color:#444}label{margin-right:10px;cursor:pointer;white-space:nowrap}button{padding:4px 10px}</style></head><body>
<header><b>築年の目視ラベル（__YEAR__年と現在の航空写真）</b><br>
<span class="meta">左: <b>__YEAR__年度</b>の写真 / 右: <b>現在</b>の写真（黄枠は同じ建物の位置）。左の黄枠の中に<b>建物があったか</b>を選んでください。<br>
<b>あり</b>=同じ場所に建物が写っている(屋根の色が違っても可) / <b>建て替え</b>=別の形・大きさの建物だった /
<b>なし</b>=空き地・畑・駐車場・森など / <b>不明</b>=雲・影・工事中などで判断できない</span>
<div id="sum"></div><button onclick="dl()">ラベルをCSV保存</button></header>
<div class="grid" id="g"></div>
<script>
const D=__DATA__,KEY="age_label___YEAR__";let L=JSON.parse(localStorage.getItem(KEY)||"{}");const g=document.getElementById("g");
D.forEach((it,i)=>{const c=document.createElement("div");c.className="card";
 const u=`https://maps.gsi.go.jp/#19/${it.lat}/${it.lon}/&base=seamlessphoto`;
 c.innerHTML=`<img loading="lazy" src="${it.src}"><div class="meta">#${i+1} ${it.town} / ${it.area}㎡ / <a href="${u}" target="_blank">地理院地図</a></div>
 <div>${[["present","あり"],["rebuilt","建て替え"],["absent","なし"],["unsure","不明"]].map(([v,t])=>`<label><input type="radio" name="${it.bid}" value="${v}" ${L[it.bid]==v?"checked":""}>${t}</label>`).join("")}</div>`;
 c.querySelectorAll("input").forEach(r=>r.onchange=()=>{L[it.bid]=r.value;localStorage.setItem(KEY,JSON.stringify(L));sum();});g.appendChild(c)});
function sum(){const v=Object.values(L);const n=k=>v.filter(x=>x===k).length;
 document.getElementById("sum").innerHTML=`確認 ${v.length}/${D.length} | あり ${n("present")} / 建て替え ${n("rebuilt")} / なし ${n("absent")} / 不明 ${n("unsure")}`}
function dl(){let s="bid,stratum,label\n"+D.filter(i=>L[i.bid]).map(i=>`${i.bid},${i.stratum},${L[i.bid]}`).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download="age_labels.csv";a.click()}
sum();</script></body></html>"""
