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
          flat_other: float = 0.0, roof_note: str = "") -> Path:
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
    data = {"munis": munis, "geo": {"type": "FeatureCollection", "features": feats}, "pts": pts,
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
@media(max-width:800px){#app{grid-template-columns:1fr;grid-template-rows:auto 60vh}#side{max-height:50vh;border-right:0}}
</style></head><body><div id="app"><div id="side">
<h1>北海道 対象戸数マップ</h1>
<div class="sub">築20年以内 / 戸建て / 持ち家 / 太陽光なし / 陸屋根を除く （統計からの推計値）</div>
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
 $("funnel").innerHTML=rows.map(([l,v])=>`<div><span>${l}</span><span class="n">${fmt(v)}</span></div><div class="bar"><i style="width:${T.det?100*v/T.det:0}%"></i></div>`).join("");
 $("fsv").textContent=$("fs").value+"%";$("fov").textContent=$("fo").value+"%"}
const fmtR=R=>R>=1000?(R/1000).toFixed(R>=10000?0:1)+" km":R+" m";
function setCenter(ll){center=ll;drawCircle();summary()}
function drawCircle(){if(circle)map.removeLayer(circle);if(!center)return;circle=L.circle(center,{radius:+RAD($("r").value),color:"#d62728",weight:2,fillOpacity:.08}).addTo(map)}
map.on("click",e=>{if(!e.originalEvent._p)setCenter(e.latlng)});
$("r").oninput=()=>{$("rv").textContent=fmtR(+RAD($("r").value));drawCircle();summary()};$("rv").textContent=fmtR(+RAD($("r").value));
$("clr").onclick=()=>{center=null;if(circle)map.removeLayer(circle);circle=null;summary()};
$("fs").oninput=$("fo").oninput=draw;$("op").oninput=()=>{$("opv").textContent=$("op").value+"%";layer.setStyle({fillOpacity:+$("op").value/100,opacity:$("op").value>0?1:0})};$("opv").textContent=$("op").value+"%";$("metric").onchange=e=>{metric=e.target.value;draw()};
$("rank").onclick=e=>{const tr=e.target.closest("tr[data-i]");if(!tr)return;const f=layer.getLayers().find(l=>l.feature.properties.i==tr.dataset.i);if(f)map.fitBounds(f.getBounds())};
$("csv").onclick=()=>{const s="市区町村,一戸建,持ち家,築20年以内,太陽光なし,陸屋根を除く対象戸数\n"+M.map(m=>{const c=calc(m,m.det);return[m.name,c.det,c.own,c.nw,c.ns,c.fin].map((v,k)=>k?Math.round(v):v).join(",")}).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+s],{type:"text/csv"}));a.download="hokkaido_target_by_municipality.csv";a.click()};
L.control.layers({"空中写真":sat,"淡色地図":pale,"標準地図":std},{}).addTo(map);
draw();
</script></body></html>"""
