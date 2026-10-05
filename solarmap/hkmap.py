"""北海道 対象戸数マップ(単一 HTML)。市町村の色分け・内訳・円(半径)内の件数集計・陸屋根率スライダー。"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS_AREA, CRS_GEO

SAPPORO_WARDS = {f"011{i:02d}" for i in range(1, 11)}


def _points(units: gpd.GeoDataFrame, muni_index: dict, seed: int = 0) -> list:
    """区画ごとの一戸建数を、面積に応じて 1〜25 点へ分けた [lon, lat, 重み, 市区町村idx] の列にする。"""
    rng = np.random.default_rng(seed)
    area_km2 = units.to_crs(CRS_AREA).area.to_numpy() / 1e6
    out = []
    for geom, det, m, a in zip(units.geometry, units["detached"], units["muni"], area_km2):
        if not det or det <= 0:
            continue
        n = int(np.clip(np.ceil(a / 1.0), 1, 25))
        if n == 1:
            p = geom.representative_point()
            pts = [(p.x, p.y)]
        else:
            minx, miny, maxx, maxy = geom.bounds
            pts = []
            for _ in range(400):
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
                if geom.contains(__import__("shapely").geometry.Point(x, y)):
                    pts.append((x, y))
                if len(pts) >= n:
                    break
            if not pts:
                p = geom.representative_point()
                pts = [(p.x, p.y)]
        w = round(float(det) / len(pts), 2)
        out.extend([round(x, 4), round(y, 4), w, muni_index[m]] for x, y in pts)
    return out


def build(units: gpd.GeoDataFrame, est: pd.DataFrame, out: Path, flat_sapporo: float = 0.0,
          flat_other: float = 0.0, roof_note: str = "", pins: pd.DataFrame | None = None,
          pins_name: str = "") -> Path:
    est = est.reset_index(drop=True)
    idx = {m: i for i, m in enumerate(est["muni"])}
    # 市町村ポリゴン（区画を融合して簡略化）
    g = units[["muni", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(0.0004, preserve_topology=True)
    g = g.dissolve(by="muni").reset_index()
    g["geometry"] = g.geometry.simplify(0.0008, preserve_topology=True)
    feats = []
    for r in g.itertuples():
        if r.muni in idx:
            feats.append({"type": "Feature", "properties": {"i": idx[r.muni]},
                          "geometry": json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"]})
    munis = [{"code": r.muni, "name": r.city_name, "det": round(float(r.detached)),
              "own": round(float(r.own_share), 4), "new": round(float(r.new_share), 4),
              "sol": round(float(r.s_new_det), 4), "nbr": round(float(r.nbr), 5),
              "soln": round(float(r.s_new11_det), 4), "nb": round(float(r.new_builds)), "sap": r.muni in SAPPORO_WARDS,
              "rural": bool(r.is_rural_default)} for r in est.itertuples()]
    pts = _points(units, idx)
    rp = units.geometry.representative_point()
    places = [[f"{c}{n}", round(x, 4), round(y, 4)] for c, n, x, y in
              zip(units["city_name"].fillna(""), units["S_NAME"].fillna(""), rp.x, rp.y)]
    pin_data = None
    if pins is not None and len(pins):
        pn = pins.sort_values("bid").reset_index(drop=True)
        pin_data = {"name": pins_name,
                    "ids": [b for b in pn["bid"]],
                    "pts": [[round(r.lon, 5), round(r.lat, 5), int(round(r.score * 100)), int(round(r.area_m2)),
                             r.S_NAME] for r in pn.itertuples()]}
    data = {"places": places, "pins": pin_data, "munis": munis, "geo": {"type": "FeatureCollection", "features": feats}, "pts": pts,
            "flat": {"sap": flat_sapporo, "other": flat_other}, "roofNote": roof_note}
    html = _HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>北海道 対象戸数マップ</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<style>
:root{--bg:#fff;--fg:#1c2733;--mut:#5b6b7a;--line:#d5dce3;--acc:#1f6feb}
@media(prefers-color-scheme:dark){:root{--bg:#161b22;--fg:#e6edf3;--mut:#9aa7b4;--line:#2d3640;--acc:#58a6ff}}
html,body{height:100%;margin:0;font:14px/1.5 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif;background:var(--bg);color:var(--fg)}
#app{display:grid;grid-template-columns:360px 1fr;height:100%}
#side{overflow:auto;padding:14px 16px;border-right:1px solid var(--line);box-sizing:border-box}
#map{height:100%}
h1{font-size:17px;margin:0 0 4px}.sub{color:var(--mut);font-size:12px;margin-bottom:10px}
.big{font-size:30px;font-weight:700;line-height:1.1}.unit{font-size:13px;font-weight:400;color:var(--mut)}
.box{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.funnel div{display:flex;justify-content:space-between;padding:2px 0}.funnel .n{font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--line);border-radius:3px;margin:1px 0 4px}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
label{display:block;font-size:12px;color:var(--mut);margin-top:6px}input[type=range]{width:100%}
.row{display:flex;gap:8px;align-items:center}.row b{min-width:48px;text-align:right}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:3px 4px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{cursor:pointer;color:var(--mut);position:sticky;top:0;background:var(--bg)}
#rank{max-height:260px;overflow:auto}.note{font-size:11px;color:var(--mut)}
button{padding:5px 10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;cursor:pointer}
.legend{background:var(--bg);color:var(--fg);padding:8px 10px;border:1px solid var(--line);border-radius:6px;font-size:12px}
.legend i{display:inline-block;width:14px;height:12px;margin-right:6px;vertical-align:-1px}
.sbox{position:relative;margin:6px 0 10px}.sbox input[type=text]{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font-size:14px}
#sres{position:absolute;left:0;right:0;top:100%;z-index:1000;background:var(--bg);border:1px solid var(--line);border-radius:8px;max-height:300px;overflow:auto;display:none;box-shadow:0 4px 14px rgba(0,0,0,.25)}
#sres div{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px}#sres div:hover,#sres div.on{background:rgba(31,111,235,.15)}#sres small{color:var(--mut);margin-left:6px}
.upin{width:24px;height:24px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);border:2px solid #fff;box-shadow:0 1px 5px rgba(0,0,0,.55);position:relative}
.upin:after{content:"";position:absolute;left:6px;top:6px;width:8px;height:8px;border-radius:50%;background:#fff}
.pp input[type=text],.pp textarea{width:100%;box-sizing:border-box;margin:2px 0;padding:4px;border:1px solid #bbb;border-radius:4px;font:13px system-ui}.pp .cols span{display:inline-block;width:20px;height:20px;border-radius:50%;margin-right:4px;cursor:pointer;border:2px solid #fff;box-shadow:0 0 0 1px #999}
.pp .cols span.on{box-shadow:0 0 0 2px #000}.pp button{margin:3px 4px 0 0;padding:3px 8px}.pp a{color:#1f6feb}
#pins .it{display:flex;justify-content:space-between;gap:6px;padding:4px 0;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px}#pins .it i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.mode-on{background:#1f6feb!important;color:#fff!important}
@media(max-width:800px){#app{grid-template-columns:1fr;grid-template-rows:auto 60vh}#side{max-height:50vh;border-right:0}}
</style></head><body><div id="app"><div id="side">
<h1>北海道 対象戸数マップ</h1>
<div class="sub">築20年以内 / 戸建て / 持ち家 / 太陽光なし / 陸屋根を除く （統計からの推計値）</div>
<div class="sbox"><input type="text" id="q" placeholder="地名・住所・施設名で検索（例: 手稲区前田一条、イオン手稲）" autocomplete="off"><div id="sres"></div></div>
<div class="row" style="margin-bottom:8px"><button id="loc">現在地</button><button id="pinmode">📍 ピンを立てる</button><span class="note" id="pinhint"></span></div>
<details class="box" id="mypins" open><summary><b>マイピン <span id="pincount">(0)</span></b></summary>
<div id="pins"></div><div class="row" style="margin-top:6px;flex-wrap:wrap"><button id="pcsv2">CSV</button><button id="pkml">KML(Googleマイマップ用)</button><button id="pimp">読み込み</button><button id="pdel">全削除</button><input type="file" id="pfile" accept=".json,.csv,.kml" style="display:none"></div>
<div class="note">ピンは<b>このブラウザに保存</b>されます（消すとき以外は残る）。他の人と共有するには CSV/KML で書き出してください。</div></details>
<div class="box"><div class="unit">対象戸数（<span id="scope">北海道全体</span>）</div>
<div class="big"><span id="total">-</span><span class="unit"> 戸</span></div>
<div class="funnel" id="funnel" style="margin-top:8px"></div></div>
<div class="box"><b>陸屋根の割合（無落雪屋根を含む）</b>
<div class="note" id="roofnote"></div>
<label>札幌市</label><div class="row"><input id="fs" type="range" min="0" max="90" step="1"><b id="fsv"></b></div>
<label>札幌市以外</label><div class="row"><input id="fo" type="range" min="0" max="90" step="1"><b id="fov"></b></div></div>
<div class="box"><b>色分けの濃さ</b>
<div class="row"><input id="op" type="range" min="0" max="90" step="5" value="40"><b id="opv"></b></div>
<div class="note">下げると空中写真がよく見えます。0%で色分けを消します。</div></div>
<div class="box" id="pinbox" style="display:none"><b>家ごとのピン（<span id="pinname"></span>のみ・試作）</b>
<div class="note">ピンは<b>対象条件の判定ではありません</b>。戸建てサイズの建物と、AIが「パネルあり」と判定した家です（的中率は低く、確認用の候補）。</div>
<label><input type="checkbox" id="pc" checked> 戸建て候補（点）<span id="pcn" class="note"></span></label>
<label><input type="checkbox" id="pa" checked> AIがパネルありと判定（赤ピン）<span id="pan" class="note"></span></label>
<label>AIスコア <span id="ptv"></span> 以上</label><input id="pt" type="range" min="10" max="90" step="5" value="50">
<div class="row" style="margin-top:6px"><button id="pgo">手稲区へ移動</button><button id="pcsv">確認ラベルをCSV保存</button></div>
<div class="note" id="plab"></div>
<div class="note">ピンを押すと写真リンクと「あり/なし」の記録欄が出ます。記録はこのブラウザに保存されます。</div></div>
<details class="box"><summary><b>この数字の出し方</b></summary>
<div class="note" style="line-height:1.6">
<b>円内の戸建て数</b>: 国勢調査2020の町丁目別「一戸建」世帯数を、区画の面積に応じた点に分けて按分し、円に入る点を合計。<br>
<b>→ 持ち家</b>: 住宅・土地統計調査2023の「持ち家一戸建 ÷ 一戸建」(市区町村別)を掛ける。<br>
<b>→ 築20年以内</b>: 持ち家一戸建のうち2006年以降の築の割合(市区町村別)を掛け、2023.10以降の新築(着工統計)を足す。<br>
<b>→ 太陽光なし</b>: 築20年以内の住宅の太陽光設置率(市区町村別・北海道平均へ補正)を引く。<br>
<b>→ 陸屋根を除く</b>: 屋根ラベルの実測の陸屋根率(上のスライダー)を引く。<br>
各行の右の%は、ひとつ前の行に対する割合です。</div></details>
<div class="box"><b>円で集計</b>
<div class="note">地図をクリックすると中心を置きます。</div>
<label>半径 <span id="rv"></span></label><input id="r" type="range" min="0" max="100" step="1" value="40">
<div class="row" style="margin-top:6px"><button id="clr">円を消す</button><span class="note" id="circinfo"></span></div></div>
<div class="box"><div class="row" style="justify-content:space-between"><b>市区町村ランキング</b>
<span><select id="metric"><option value="n">対象戸数</option><option value="share">対象率(対象/一戸建)</option></select></span></div>
<div id="rank"></div><div style="margin-top:6px"><button id="csv">CSVを保存</button></div></div>
<div class="note">出典: 国勢調査2020(一戸建世帯数, 町丁目別) / 住宅・土地統計調査2023(持ち家・築年・太陽光の市区町村別表) / 国土地理院(地図)。
町村は個別の表が無いため「北海道−市の合計」(町村部)の割合で代用。築20年は2006年以降の築。2023年10月以降の新築は建築着工統計(2024年実績。2025年以降は同水準と仮定)で加算。
円内の件数は町丁目内を点に分けて按分した概算で、半径が小さい(約500m未満)と誤差が大きい。</div>
</div><div id="map"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const D=__DATA__;const M=D.munis,P=D.pts;
const $=id=>document.getElementById(id);const fmt=n=>Math.round(n).toLocaleString("ja-JP");
$("fs").value=Math.round(D.flat.sap*100);$("fo").value=Math.round(D.flat.other*100);$("roofnote").textContent=D.roofNote||"";
const flatOf=m=>(m.sap?+$("fs").value:+$("fo").value)/100;
function calc(m,det){const own=det*m.own,nw1=own*m.new,nw2=det*m.nbr,nw=nw1+nw2,ns=nw1*(1-m.sol)+nw2*(1-m.soln),fin=ns*(1-flatOf(m));return{det,own,nw,nw2,ns,fin}}
const map=L.map("map",{preferCanvas:true}).setView([43.4,142.6],7);
const GSI='<a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>';
const pale=L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",{maxZoom:19,maxNativeZoom:18,attribution:GSI});
const std=L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",{maxZoom:19,maxNativeZoom:18,attribution:GSI});
const sat=L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",{maxZoom:19,maxNativeZoom:18,attribution:GSI}).addTo(map);
let metric="n",layer,dots,circle,center=null;
const RAD=v=>Math.round(100*Math.pow(10,v/100*2.3));  // 100m〜20km
function rank(){const rows=M.map((m,i)=>{const c=calc(m,m.det);return{i,name:m.name,n:c.fin,det:m.det,share:m.det?c.fin/m.det:0}});return rows}
function color(v,max){const t=Math.min(1,Math.sqrt(v/max));const a=[255,247,200],b=[16,80,160];
 return"rgb("+a.map((x,k)=>Math.round(x+(b[k]-x)*t)).join(",")+")"}
function value(r){return metric==="n"?r.n:r.share}
function draw(){const rows=rank(),max=Math.max(...rows.map(value));if(layer)map.removeLayer(layer);
 layer=L.geoJSON(D.geo,{style:f=>({color:"#ffffff",weight:.8,fillOpacity:+$("op").value/100,fillColor:color(value(rows[f.properties.i]),max)}),
 onEachFeature:(f,l)=>{const r=rows[f.properties.i];l.bindTooltip(M[r.i].name+"　"+(metric==="n"?fmt(r.n)+"戸":(r.share*100).toFixed(1)+"%"));
 l.on("click",e=>{if(e.originalEvent.shiftKey||!$("circ")){} popup(l,r.i,e.latlng)})}}).addTo(map);
 layer.bringToBack();legend(max);table(rows);summary()}
function popup(l,i,ll){const m=M[i],c=calc(m,m.det);
 L.popup().setLatLng(ll).setContent(`<b>${m.name}</b>${m.rural?'<br><span style="font-size:11px">町村部の割合で推計</span>':""}<br>一戸建 ${fmt(c.det)}<br>持ち家 ${fmt(c.own)}<br>築20年以内 ${fmt(c.nw)}（うち2023.10以降の新築 ${fmt(c.nw2)}）<br>太陽光なし ${fmt(c.ns)}<br><b>陸屋根を除く ${fmt(c.fin)}戸</b>`).openOn(map);
 setCenter(ll)}
let lg;function legend(max){if(lg)map.removeControl(lg);lg=L.control({position:"bottomright"});lg.onAdd=()=>{const d=L.DomUtil.create("div","legend");
 const f=metric==="n"?fmt:(v=>(v*100).toFixed(0)+"%");d.innerHTML=[0,.1,.25,.5,.75,1].map(t=>`<div><i style="background:${color(t*t*max,max)}"></i>${f(t*t*max)}${t===1?"以上":""}</div>`).join("");return d};lg.addTo(map)}
function table(rows){const s=[...rows].sort((a,b)=>value(b)-value(a)).slice(0,40);
 $("rank").innerHTML="<table><tr><th>市区町村</th><th>対象</th><th>一戸建</th></tr>"+s.map(r=>`<tr data-i="${r.i}"><td>${r.name}</td><td>${metric==="n"?fmt(r.n):(r.share*100).toFixed(1)+"%"}</td><td>${fmt(r.det)}</td></tr>`).join("")+"</table>"}
function summary(){let T={det:0,own:0,nw:0,ns:0,fin:0};
 if(center){const R=+RAD($("r").value),acc={};const cl=Math.cos(center.lat*Math.PI/180);
  for(let k=0;k<P.length;k+=1){const p=P[k];const dx=(p[0]-center.lng)*111320*cl,dy=(p[1]-center.lat)*110540;if(dx*dx+dy*dy<=R*R){acc[p[3]]=(acc[p[3]]||0)+p[2]}}
  for(const i in acc){const c=calc(M[i],acc[i]);for(const k in T)T[k]+=c[k]}
  $("scope").textContent="円内 半径 "+fmtR(R);$("circinfo").textContent="一戸建 "+fmt(T.det)+" 戸";
 }else{M.forEach(m=>{const c=calc(m,m.det);for(const k in T)T[k]+=c[k]});$("scope").textContent="北海道全体";$("circinfo").textContent=""}
 $("total").textContent=fmt(T.fin);
 const rows=[["戸建て",T.det],["持ち家(賃貸を除く)",T.own],["築20年以内(新築含む)",T.nw],["太陽光なし",T.ns],["陸屋根を除く",T.fin]];
 $("funnel").innerHTML=rows.map(([l,v],k)=>`<div><span>${l}${k?` <span class="note">(${rows[k-1][1]?(100*v/rows[k-1][1]).toFixed(1):"-"}%)</span>`:""}</span><span class="n">${fmt(v)}</span></div><div class="bar"><i style="width:${T.det?100*v/T.det:0}%"></i></div>`).join("");
 $("fsv").textContent=$("fs").value+"%";$("fov").textContent=$("fo").value+"%"}
const fmtR=R=>R>=1000?(R/1000).toFixed(R>=10000?0:1)+" km":R+" m";
function setCenter(ll){center=ll;drawCircle();summary()}
function drawCircle(){if(circle)map.removeLayer(circle);if(!center)return;circle=L.circle(center,{radius:+RAD($("r").value),color:"#d62728",weight:2,fillOpacity:.08}).addTo(map)}
map.on("click",e=>{if(pinMode){addPin(e.latlng.lat,e.latlng.lng);return}if(pinClick(e))return;setCenter(e.latlng)});
$("r").oninput=()=>{$("rv").textContent=fmtR(+RAD($("r").value));drawCircle();summary()};$("rv").textContent=fmtR(+RAD($("r").value));
$("clr").onclick=()=>{center=null;if(circle)map.removeLayer(circle);circle=null;summary()};
$("fs").oninput=$("fo").oninput=draw;$("op").oninput=()=>{$("opv").textContent=$("op").value+"%";layer.setStyle({fillOpacity:+$("op").value/100,opacity:$("op").value>0?1:0})};$("opv").textContent=$("op").value+"%";$("metric").onchange=e=>{metric=e.target.value;draw()};
$("rank").onclick=e=>{const tr=e.target.closest("tr[data-i]");if(!tr)return;const f=layer.getLayers().find(l=>l.feature.properties.i==tr.dataset.i);if(f)map.fitBounds(f.getBounds())};
$("csv").onclick=()=>{const s="市区町村,一戸建,持ち家,築20年以内,太陽光なし,陸屋根を除く対象戸数\n"+M.map(m=>{const c=calc(m,m.det);return[m.name,c.det,c.own,c.nw,c.ns,c.fin].map((v,k)=>k?Math.round(v):v).join(",")}).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+s],{type:"text/csv"}));a.download="hokkaido_target_by_municipality.csv";a.click()};
L.control.layers({"空中写真":sat,"淡色地図":pale,"標準地図":std},{}).addTo(map);
draw();
/* ---------------- 検索・マイピン ---------------- */
const COLS=["#d62728","#1f77b4","#2ca02c","#ff7f0e","#9467bd"];
let UP=[];try{UP=JSON.parse(localStorage.getItem("user_pins_v1")||"[]")}catch(e){}
const saveUP=()=>{try{localStorage.setItem("user_pins_v1",JSON.stringify(UP))}catch(e){}};
const esc=t=>String(t||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function around(lat,lng,R){const acc={},cl=Math.cos(lat*Math.PI/180);
 for(let k=0;k<P.length;k++){const p=P[k],dx=(p[0]-lng)*111320*cl,dy=(p[1]-lat)*110540;if(dx*dx+dy*dy<=R*R)acc[p[3]]=(acc[p[3]]||0)+p[2]}
 const T={det:0,own:0,nw:0,ns:0,fin:0};for(const i in acc){const c=calc(M[i],acc[i]);for(const k in T)T[k]+=c[k]}return T}
const gmapsDir=(lat,lng)=>`https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`;
const gmapsAt=(lat,lng)=>`https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
function aroundBox(lat,lng){const box=document.createElement("div");const sel=document.createElement("select");
 [300,500,1000,2000].forEach(r=>{const o=document.createElement("option");o.value=r;o.textContent="半径 "+(r>=1000?r/1000+" km":r+" m");if(r===500)o.selected=true;sel.appendChild(o)});
 const out=document.createElement("div");const upd=()=>{const T=around(lat,lng,+sel.value);out.innerHTML=`<b>対象 ${fmt(T.fin)} 戸</b> <span style="color:#666">（一戸建 ${fmt(T.det)}）</span>`};
 sel.onchange=upd;box.appendChild(sel);box.appendChild(out);upd();return box}
let tmpMarker=null,mePos=null,pinMode=false;
function iconOf(c){return L.divIcon({className:"",html:`<div class="upin" style="background:${c}"></div>`,iconSize:[24,24],iconAnchor:[6,27],popupAnchor:[6,-26]})}
const upLayer=L.layerGroup().addTo(map);let upMk={};
function pinPopup(pin){const d=document.createElement("div");d.className="pp";d.style.minWidth="220px";
 d.innerHTML=`<input type="text" class="n" placeholder="名前" value="${esc(pin.name)}"><textarea class="m" rows="2" placeholder="メモ">${esc(pin.memo)}</textarea><div class="cols"></div><div class="ar"></div>
 <div style="margin:4px 0"><a target="_blank" href="${gmapsDir(pin.lat,pin.lng)}">経路(Googleマップ)</a> ・ <a target="_blank" href="${gmapsAt(pin.lat,pin.lng)}">Googleマップで開く</a> ・ <a target="_blank" href="https://maps.gsi.go.jp/#19/${pin.lat}/${pin.lng}/&base=seamlessphoto">地理院地図</a></div>
 <div class="note" style="color:#666">${pin.lat.toFixed(5)}, ${pin.lng.toFixed(5)}</div><button class="sv">保存</button><button class="dl">削除</button>`;
 const cols=d.querySelector(".cols");COLS.forEach(c=>{const sp=document.createElement("span");sp.style.background=c;if(c===pin.color)sp.className="on";sp.onclick=()=>{pin.color=c;cols.querySelectorAll("span").forEach(x=>x.className="");sp.className="on"};cols.appendChild(sp)});
 d.querySelector(".ar").appendChild(aroundBox(pin.lat,pin.lng));
 d.querySelector(".sv").onclick=()=>{pin.name=d.querySelector(".n").value||"ピン";pin.memo=d.querySelector(".m").value;saveUP();renderUP();map.closePopup()};
 d.querySelector(".dl").onclick=()=>{UP=UP.filter(x=>x.id!==pin.id);saveUP();renderUP();map.closePopup()};return d}
function renderUP(){upLayer.clearLayers();upMk={};UP.forEach(pin=>{const m=L.marker([pin.lat,pin.lng],{icon:iconOf(pin.color),title:pin.name}).addTo(upLayer);
 m.bindPopup(()=>pinPopup(pin),{maxWidth:300});upMk[pin.id]=m});
 $("pincount").textContent="("+UP.length+")";
 $("pins").innerHTML=UP.map(p=>`<div class="it" data-id="${p.id}"><span><i style="background:${p.color}"></i>${esc(p.name)}</span><span class="note">${esc((p.memo||"").slice(0,16))}</span></div>`).join("")||'<div class="note">まだありません。「ピンを立てる」を押して地図をクリック、または検索結果から立てられます。</div>'}
$("pins").onclick=e=>{const it=e.target.closest(".it");if(!it)return;const p=UP.find(x=>x.id===it.dataset.id);if(p){map.setView([p.lat,p.lng],Math.max(map.getZoom(),16));upMk[p.id].openPopup()}};
function addPin(lat,lng,name){const pin={id:"u"+Date.now().toString(36)+Math.floor(Math.random()*1e3),lat:+lat.toFixed(6),lng:+lng.toFixed(6),name:name||"ピン"+(UP.length+1),memo:"",color:COLS[UP.length%COLS.length]};
 UP.push(pin);saveUP();renderUP();upMk[pin.id].openPopup();return pin}
$("pinmode").onclick=()=>{pinMode=!pinMode;$("pinmode").classList.toggle("mode-on",pinMode);$("pinhint").textContent=pinMode?"地図をクリックで追加（もう一度押すと解除）":"";map.getContainer().style.cursor=pinMode?"crosshair":""};
map.on("contextmenu",e=>addPin(e.latlng.lat,e.latlng.lng));   // 右クリック/長押しでも立てられる
const dl=(name,txt,type)=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\ufeff"+txt],{type}));a.download=name;a.click()};
$("pcsv2").onclick=()=>dl("my_pins.csv","name,lat,lng,memo,color\n"+UP.map(p=>[p.name,p.lat,p.lng,p.memo,p.color].map(v=>'"'+String(v||"").replace(/"/g,'""')+'"').join(",")).join("\n"),"text/csv");
$("pkml").onclick=()=>dl("my_pins.kml",`<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>マイピン</name>`+UP.map(p=>`<Placemark><name>${esc(p.name)}</name><description>${esc(p.memo)}</description><Point><coordinates>${p.lng},${p.lat},0</coordinates></Point></Placemark>`).join("")+`</Document></kml>`,"application/vnd.google-earth.kml+xml");
$("pimp").onclick=()=>$("pfile").click();
$("pfile").onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{let add=[];const t=r.result.replace(/^\ufeff/,"");
  if(f.name.endsWith(".kml")){const x=new DOMParser().parseFromString(t,"text/xml");x.querySelectorAll("Placemark").forEach(pm=>{const c=(pm.querySelector("coordinates")||{}).textContent;if(!c)return;const[lng,lat]=c.trim().split(",").map(Number);add.push({name:(pm.querySelector("name")||{}).textContent||"ピン",memo:(pm.querySelector("description")||{}).textContent||"",lat,lng})})}
  else{t.split(/\r?\n/).slice(1).forEach(l=>{const m=l.match(/("([^"]|"")*"|[^,]*)(,|$)/g);if(!m)return;const v=m.map(z=>z.replace(/,$/,"").replace(/^"|"$/g,"").replace(/""/g,'"'));const lat=+v[1],lng=+v[2];if(isFinite(lat)&&isFinite(lng)&&v[1]!=="")add.push({name:v[0],lat,lng,memo:v[3]||"",color:v[4]})})}
  add.forEach((a,i)=>UP.push({id:"u"+Date.now().toString(36)+i,lat:a.lat,lng:a.lng,name:a.name,memo:a.memo,color:COLS.includes(a.color)?a.color:COLS[(UP.length)%COLS.length]}));saveUP();renderUP();if(add.length)map.setView([add[0].lat,add[0].lng],14)};r.readAsText(f);e.target.value=""};
$("pdel").onclick=()=>{if(UP.length&&confirm("マイピンを全て削除しますか？")){UP=[];saveUP();renderUP()}};
$("loc").onclick=()=>{if(!navigator.geolocation){alert("この環境では現在地を取得できません");return}
 navigator.geolocation.getCurrentPosition(pos=>{const ll=[pos.coords.latitude,pos.coords.longitude];map.setView(ll,16);if(mePos)map.removeLayer(mePos);mePos=L.circleMarker(ll,{radius:8,color:"#fff",weight:2,fillColor:"#1a73e8",fillOpacity:1}).addTo(map).bindPopup("現在地")},()=>alert("現在地を取得できませんでした（ブラウザの許可、または https/対応環境が必要です）"))};
// ---- 検索: 町丁目(手元) → 住所(国土地理院) → 施設(OpenStreetMap) ----
const PL_=D.places;let sTimer=null,sSeq=0,sItems=[],sSel=-1;
function showRes(){const el=$("sres");if(!sItems.length){el.style.display="none";return}
 el.innerHTML=sItems.map((r,i)=>`<div data-i="${i}" class="${i===sSel?"on":""}">${esc(r.t)}<small>${esc(r.s)}</small></div>`).join("");el.style.display="block"}
function gotoRes(r){$("sres").style.display="none";$("q").value=r.t;const z=r.z||16;map.setView([r.lat,r.lng],z);if(tmpMarker)map.removeLayer(tmpMarker);
 const d=document.createElement("div");d.className="pp";d.style.minWidth="210px";d.innerHTML=`<b>${esc(r.t)}</b><div class="ar"></div><div style="margin:4px 0"><a target="_blank" href="${gmapsDir(r.lat,r.lng)}">ここへ経路(Googleマップ)</a></div><button class="ap">ここにピンを立てる</button>`;
 d.querySelector(".ar").appendChild(aroundBox(r.lat,r.lng));d.querySelector(".ap").onclick=()=>{map.removeLayer(tmpMarker);tmpMarker=null;addPin(r.lat,r.lng,r.t.slice(0,24))};
 tmpMarker=L.marker([r.lat,r.lng],{icon:iconOf("#1a73e8"),zIndexOffset:900}).addTo(map).bindPopup(d,{maxWidth:300}).openPopup()}
$("sres").onclick=e=>{const d=e.target.closest("div[data-i]");if(d)gotoRes(sItems[+d.dataset.i])};
$("q").onkeydown=e=>{if(e.key==="Enter"){e.preventDefault();const r=sItems[sSel>=0?sSel:0];if(r)gotoRes(r)}else if(e.key==="ArrowDown"||e.key==="ArrowUp"){e.preventDefault();sSel=(sSel+(e.key==="ArrowDown"?1:-1)+sItems.length)%Math.max(sItems.length,1);showRes()}else if(e.key==="Escape")$("sres").style.display="none"};
document.addEventListener("click",e=>{if(!e.target.closest(".sbox"))$("sres").style.display="none"});
$("q").oninput=()=>{clearTimeout(sTimer);sTimer=setTimeout(doSearch,250)};
async function jget(u,sig){const r=await fetch(u,{signal:sig});if(!r.ok)throw new Error(r.status);return r.json()}
async function doSearch(){const q=$("q").value.trim(),seq=++sSeq;if(q.length<1){sItems=[];showRes();return}
 let items=[];const m=q.match(/^(-?\d{1,3}(?:\.\d+)?)\s*[,、 ]\s*(-?\d{1,3}(?:\.\d+)?)$/);
 if(m){const a=+m[1],b=+m[2];const[lat,lng]=Math.abs(a)<=90&&b>90?[a,b]:[b,a];items.push({t:"座標 "+lat+", "+lng,s:"緯度,経度",lat,lng,z:17})}
 const qn=q.replace(/[\s　]+/g,"");
 const loc=PL_.filter(p=>p[0].includes(qn)).slice(0,6);loc.forEach(p=>items.push({t:p[0],s:"町丁目",lat:p[2],lng:p[1],z:15}));
 sItems=items;sSel=-1;showRes();if(q.length<2)return;
 const ac=new AbortController();setTimeout(()=>ac.abort(),6000);
 const [g,n]=await Promise.allSettled([jget("https://msearch.gsi.go.jp/address-search/AddressSearch?q="+encodeURIComponent(q),ac.signal),
  jget("https://nominatim.openstreetmap.org/search?format=jsonv2&limit=8&countrycodes=jp&accept-language=ja&viewbox=139.3,45.6,145.9,41.3&bounded=1&q="+encodeURIComponent(q),ac.signal)]);
 if(seq!==sSeq)return;
 if(g.status==="fulfilled")g.value.filter(f=>f.properties.title.startsWith("北海道")).slice(0,6).forEach(f=>{const t=f.properties.title,c=f.geometry.coordinates;if(!items.some(i=>i.t===t))items.push({t,s:"住所",lat:c[1],lng:c[0],z:17})});
 if(n.status==="fulfilled")n.value.slice(0,6).forEach(f=>{const t=(f.name||f.display_name.split(",")[0]);items.push({t,s:"施設・地名 "+(f.display_name||"").split(",").slice(1,3).join(" ").slice(0,24),lat:+f.lat,lng:+f.lon,z:17})});
 if(!items.length)items.push({t:"北海道内で見つかりませんでした",s:"",lat:map.getCenter().lat,lng:map.getCenter().lng,z:map.getZoom()});
 sItems=items;showRes()}
renderUP();
/* ---------------- 家ごとのピン ---------------- */
const PD=D.pins;let LAB={};try{LAB=JSON.parse(localStorage.getItem("pin_labels_v1")||"{}")}catch(e){}
const saveLab=()=>{try{localStorage.setItem("pin_labels_v1",JSON.stringify(LAB))}catch(e){}};
if(PD){
 $("pinbox").style.display="";$("pinname").textContent=PD.name;
 const N=PD.pts.length,thr=()=>+$("pt").value/100;
 const col=i=>LAB[PD.ids[i]]==="yes"?"#2ca02c":LAB[PD.ids[i]]==="no"?"#8a8a8a":LAB[PD.ids[i]]==="unsure"?"#ff9800":null;
 const PL=L.Layer.extend({
  onAdd(m){this._m=m;const c=this._c=L.DomUtil.create("canvas");c.style.position="absolute";c.style.pointerEvents="none";c.style.zIndex=450;m.getPanes().overlayPane.appendChild(c);m.on("moveend zoomend resize",this.draw,this);this.draw()},
  onRemove(m){m.off("moveend zoomend resize",this.draw,this);this._c.remove()},
  draw(){const m=this._m,s=m.getSize(),c=this._c;L.DomUtil.setPosition(c,m.containerPointToLayerPoint([0,0]));c.width=s.x;c.height=s.y;
   const g=c.getContext("2d");g.clearRect(0,0,s.x,s.y);if(m.getZoom()<14)return;const b=m.getBounds();const r=m.getZoom()>=17?4:m.getZoom()>=16?3:2;
   for(let i=0;i<N;i++){const p=PD.pts[i];if(p[1]<b.getSouth()||p[1]>b.getNorth()||p[0]<b.getWest()||p[0]>b.getEast())continue;
    const q=m.latLngToContainerPoint([p[1],p[0]]);const cc=col(i);g.beginPath();g.arc(q.x,q.y,cc?r+1:r,0,6.283);g.fillStyle=cc||"rgba(255,255,255,.85)";g.fill();g.lineWidth=.6;g.strokeStyle="rgba(0,0,0,.5)";g.stroke()}}
 });
 const dots=new PL();dots.addTo(map);let red=L.layerGroup().addTo(map);window.__red=red;
 function popupHtml(i){const p=PD.pts[i],id=PD.ids[i],t=LAB[id]||"";
  return `<b>${id}</b> ${p[4]||""}<br>建築面積 約${p[3]}㎡ / AIスコア ${(p[2]/100).toFixed(2)}<br>
  <a target="_blank" href="https://maps.gsi.go.jp/#19/${p[1]}/${p[0]}/&base=seamlessphoto">地理院地図で見る</a><br>
  ${[["yes","パネルあり"],["no","なし"],["unsure","不明"]].map(([v,l])=>`<label style="display:inline;margin-right:8px"><input type="radio" name="pl" data-id="${id}" value="${v}" ${t===v?"checked":""}> ${l}</label>`).join("")}`}
 function showPin(i){const p=PD.pts[i];L.popup({offset:[0,-4]}).setLatLng([p[1],p[0]]).setContent(popupHtml(i)).openOn(map)}
 map.on("popupopen",e=>e.popup.getElement().querySelectorAll('input[name="pl"]').forEach(r=>r.onchange=()=>{LAB[r.dataset.id]=r.value;saveLab();dots.draw();redraw();labsum()}));
 function redraw(){red.clearLayers();let n=0;const t=thr()*100;
  if($("pa").checked)for(let i=0;i<N;i++){const p=PD.pts[i];if(p[2]<t)continue;n++;
   L.circleMarker([p[1],p[0]],{radius:6,color:"#fff",weight:1.5,fillColor:col(i)||"#d62728",fillOpacity:1,bubblingMouseEvents:false}).on("click",()=>showPin(i)).addTo(red)}
  $("pan").textContent=" "+n+"件";$("ptv").textContent=(+$("pt").value/100).toFixed(2)}
 function labsum(){const v=Object.values(LAB);$("plab").textContent="記録済み: あり "+v.filter(x=>x==="yes").length+" / なし "+v.filter(x=>x==="no").length+" / 不明 "+v.filter(x=>x==="unsure").length}
 window.pinClick=e=>{if(!$("pc").checked||map.getZoom()<14)return false;let best=-1,bd=14*14;const b=map.getBounds();
  for(let i=0;i<N;i++){const p=PD.pts[i];if(p[1]<b.getSouth()||p[1]>b.getNorth()||p[0]<b.getWest()||p[0]>b.getEast())continue;
   const q=map.latLngToContainerPoint([p[1],p[0]]),dx=q.x-e.containerPoint.x,dy=q.y-e.containerPoint.y,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i}}
  if(best<0)return false;showPin(best);return true};
 $("pc").onchange=()=>{$("pc").checked?dots.addTo(map):map.removeLayer(dots)};$("pcn").textContent=" "+N.toLocaleString("ja-JP")+"棟";
 $("pa").onchange=redraw;$("pt").oninput=redraw;
 $("pgo").onclick=()=>map.setView([43.12,141.24],15);
 $("pcsv").onclick=()=>{const s="bid,lon,lat,ai_score,truth\n"+Object.keys(LAB).map(id=>{const i=PD.ids.indexOf(id);const p=PD.pts[i];return[id,p[0],p[1],(p[2]/100).toFixed(2),LAB[id]].join(",")}).join("\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download="pin_labels.csv";a.click()};
 redraw();labsum();
}else window.pinClick=()=>false;
</script></body></html>"""
