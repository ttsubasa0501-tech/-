"""CLI。各ステップは work ディレクトリ内の成果物でつながる（途中再開可）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import aggregate as agg
from . import buildings as bl
from . import age, estat, fgd, hkmap, hokkaido, jyutaku, mapout, review, roofsample, vtile
from .config import TILE_ZOOM, WARDS
from .detect import make_detector
from .tiles import TileStore, building_crop, needed_tiles


def _code(a) -> str:
    return WARDS.get(a.ward, a.ward)


def cmd_boundary(a):
    w = Path(a.work)
    shp = Path(a.file) if a.file else estat.download_boundary(_code(a), w / "raw")
    gdf = estat.load_boundary(shp)
    gdf.to_file(w / "boundary.gpkg", driver="GPKG")
    print(f"[boundary] {len(gdf)} 町丁目 -> {w/'boundary.gpkg'}")


def cmd_households(a):
    w = Path(a.work)
    csv = Path(a.file) if a.file else estat.download_stats(a.stats_id, _code(a), w / "raw")
    df = estat.load_detached_households(csv, a.keyword)
    df.to_csv(w / "households.csv", index=False)


def cmd_buildings(a):
    w = Path(a.work)
    boundary = gpd.read_file(w / "boundary.gpkg")
    if a.gml:
        raw = fgd.load_bldA([Path(p) for p in a.gml])
    else:  # 基盤地図情報 GML が無ければ国土地理院ベクトルタイル(BldA)を使う
        raw = vtile.load_bldA_vtile(boundary, w / "vtiles", workers=a.workers)
    b = bl.extract_detached_candidates(raw, boundary, a.min_area, a.max_area, a.min_compact,
                                       tuple(a.types) if a.types else None)
    b.merge(boundary[["KEY_CODE", "S_NAME"]], on="KEY_CODE").to_file(w / "buildings.gpkg", driver="GPKG")
    print(f"[buildings] 全 {len(raw)} 棟 -> 戸建て候補 {len(b)} 棟")


def _store(a) -> TileStore:
    return TileStore(Path(a.work) / "tiles", TILE_ZOOM, offline=a.offline, delay=a.delay)


def cmd_tiles(a):
    b = gpd.read_file(Path(a.work) / "buildings.gpkg")
    need = needed_tiles(b.geometry, TILE_ZOOM)
    print(f"[tiles] 必要タイル {len(need)} 枚")
    ok, ng = _store(a).prefetch(need, a.workers)
    print(f"[tiles] 新規取得 {ok} / 失敗・範囲外 {ng}")


def cmd_detect(a):
    w = Path(a.work)
    b = gpd.read_file(w / "buildings.gpkg")
    kw = {} if a.detector == "baseline" else {"conf": a.conf, "imgsz": a.imgsz, "min_overlap": a.min_overlap}
    det = make_detector(a.detector, **kw)
    if a.threshold is not None:
        det.threshold = a.threshold
    store = _store(a)
    part = w / "predictions.partial.csv"   # 途中経過。中断しても続きから再開できる
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {r["bid"] for r in rows}
    if done:
        print(f"[detect] 再開: {len(done)} 棟は判定済み")
    for i, r in enumerate(b.itertuples()):
        if r.bid in done:
            continue
        c = building_crop(r.geometry, store)
        ok = c.missing == 0
        s = det.score(c) if ok else 0.0
        rows.append({"bid": r.bid, "score": s, "pred": bool(ok and s >= det.threshold), "valid": ok})
        if len(rows) % 1000 == 0:
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"[detect] {len(rows)}/{len(b)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(w / "predictions.csv", index=False)
    (w / "detector.json").write_text(json.dumps({"name": det.name, "threshold": det.threshold}))
    print(f"[detect] {det.name}: 有効 {df.valid.sum()}/{len(df)}, パネルあり {df.pred.sum()}")


def cmd_hk_units(a):
    g = hokkaido.build_units(Path(a.work))
    print(f"[hk-units] {len(g)} 区画, 一戸建 {g.detached.sum():.0f} 世帯, {g.muni.nunique()} 市区町村")


def cmd_hk_map(a):
    w = Path(a.work)
    units = gpd.read_file(w / "units.gpkg")
    cu = units.groupby(["muni", "city_name"], as_index=False)["detached"].sum()
    F = jyutaku.build_factors(w, cu)
    est = jyutaku.assign_to_census(F, cu, jyutaku.fetch_new_builds(w))
    est.to_csv(w / "municipal_estimates.csv", index=False)
    pins = None
    if a.pins_work:   # 家ごとのピン(判定済みの区): 戸建て候補 + AI のスコア
        pw = Path(a.pins_work)
        b = gpd.read_file(pw / "buildings.gpkg")
        pr = pd.read_csv(pw / "predictions.csv")
        pins = b[["bid", "lon", "lat", "area_m2", "S_NAME"]].merge(pr[["bid", "score", "valid"]], on="bid")
        pins["score"] = pins["score"].where(pins["valid"], 0.0)
    out = hkmap.build(units, est, w / "hokkaido_map.html", a.flat_sapporo / 100, a.flat_other / 100,
                      a.roof_note, pins, a.pins_name)
    pre = est["pre_roof"].sum()
    print(f"[hk-map] {out}  陸屋根を引く前の対象 {pre:,.0f} 戸（築20年以内・新築含む・持ち家・戸建て・太陽光なし）")


def cmd_roofstats(a):
    lab = pd.read_csv(a.labels, dtype={"muni": str})
    lab["sapporo"] = lab["sapporo"].astype(str).str.lower() == "true"
    for k, v in roofsample.roof_stats(lab).items():
        print(f"{k}: 陸屋根 {v['flat']}/{v['n']} = {v['rate']*100:.0f}% (95%CI {v['lo']*100:.0f}〜{v['hi']*100:.0f}%) 判別不能 {v['unsure']}")


def cmd_age_features(a):
    w = Path(a.work)
    b = gpd.read_file(w / "buildings.gpkg")
    store = TileStore(w / f"tiles_{a.year}", url=age.OLD_URL.replace("{year}", str(a.year)), delay=0.02, ext="png")
    store.prefetch(needed_tiles(b.geometry, TILE_ZOOM, 12), 8)
    f = age.extract_features(b, w, a.year, a.procs)
    f.to_csv(w / "age_features.csv", index=False)
    print(f"[age] 特徴量 {len(f)} 棟 -> {w/'age_features.csv'}")


def cmd_age_review(a):
    w = Path(a.work)
    b = gpd.read_file(w / "buildings.gpkg")
    f = pd.read_csv(w / "age_features.csv")
    s = age.sample_for_review(f, a.n, seed=a.seed)
    page = age.build_review_page(s, b, w, a.year, w / "age_review" / "age_review.html")
    print(f"[age] {page}")


def cmd_roofsample(a):
    w = Path(a.work)
    units = gpd.read_file(w / "units.gpkg")
    s = roofsample.sample_houses(units, w / "vtiles", a.n, a.seed)
    store = TileStore(w / "tiles", TILE_ZOOM, delay=0.02)
    store.prefetch(needed_tiles(s.geometry, TILE_ZOOM, 18), 8)
    page = roofsample.build_page(s, store, w / "roof_review" / "roof_review.html")
    print(f"[roofsample] {len(s)} 軒 -> {page}")


def cmd_aggregate(a):
    w = Path(a.work)
    boundary = gpd.read_file(w / "boundary.gpkg")
    b = gpd.read_file(w / "buildings.gpkg")
    pred = pd.read_csv(w / "predictions.csv")
    if a.threshold is not None:  # 保存済みスコアからしきい値だけ変えて再集計
        pred["pred"] = pred["valid"] & (pred["score"] >= a.threshold)
    hh = pd.read_csv(w / "households.csv", dtype={"KEY_CODE": str})
    out = agg.aggregate(boundary, b, pred, hh)
    mapout.write_geojson(out, w / "solar_rate.geojson")
    pos = b.merge(pred[pred.pred], on="bid")
    mapout.write_html(out, w / "solar_rate_map.html", pos)
    print(f"[aggregate] {w/'solar_rate.geojson'}, {w/'solar_rate_map.html'}  "
          f"(設置率 中央値 {out['rate'].median():.3f})")


def cmd_review(a):
    w = Path(a.work)
    b = gpd.read_file(w / "buildings.gpkg")
    pred = pd.read_csv(w / "predictions.csv")
    meta = json.loads((w / "detector.json").read_text())
    page = review.build_review(pred, b, _store(a), w / "review", a.n, a.seed,
                               meta["name"], meta["threshold"])
    print(f"[review] {page} をブラウザで開いて目視確認 → 「ラベルをCSV保存」→ evaluate")


def cmd_evaluate(a):
    w = Path(a.work)
    pred = pd.read_csv(w / "predictions.csv")
    v = pred[pred.valid]
    totals = {"pos": int(v.pred.sum()), "neg": int((~v.pred).sum())}
    res = review.evaluate(pd.read_csv(a.labels), totals)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="solarmap")
    p.add_argument("--work", default="work/teine", help="作業ディレクトリ")
    p.add_argument("--ward", default="手稲区")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("boundary", help="1. e-Stat 小地域境界"); s.add_argument("--file")
    s.set_defaults(f=cmd_boundary)
    s = sub.add_parser("households", help="1. 一戸建世帯数(統計GIS CSV)")
    s.add_argument("--file"); s.add_argument("--stats-id", default="T001086"); s.add_argument("--keyword", default="一戸建")
    s.set_defaults(f=cmd_households)
    s = sub.add_parser("buildings", help="2. 基盤地図情報 BldA から戸建て候補抽出")
    s.add_argument("gml", nargs="*"); s.add_argument("--workers", type=int, default=4); s.add_argument("--min-area", type=float, default=50)
    s.add_argument("--max-area", type=float, default=250)
    s.add_argument("--min-compact", type=float, default=0.6)
    s.add_argument("--types", nargs="*"); s.set_defaults(f=cmd_buildings)
    for name, fn, h in (("tiles", cmd_tiles, "3. 空中写真タイル取得"),
                        ("detect", cmd_detect, "4. パネル判定"),
                        ("review", cmd_review, "目視確認ページ生成")):
        s = sub.add_parser(name, help=h)
        s.add_argument("--offline", action="store_true"); s.add_argument("--delay", type=float, default=0.05)
        s.add_argument("--workers", type=int, default=4)
        if name == "detect":
            s.add_argument("--detector", default="baseline"); s.add_argument("--threshold", type=float)
            s.add_argument("--conf", type=float, default=0.25); s.add_argument("--imgsz", type=int, default=640)
            s.add_argument("--min-overlap", type=float, default=0.1)
        if name == "review":
            s.add_argument("-n", type=int, default=100); s.add_argument("--seed", type=int, default=0)
        s.set_defaults(f=fn)
    s = sub.add_parser("hk-units", help="北海道全域の町丁目別 一戸建世帯数"); s.set_defaults(f=cmd_hk_units)
    s = sub.add_parser("hk-map", help="北海道 対象戸数マップ(要 ESTAT_APP_ID)")
    s.add_argument("--flat-sapporo", type=float, default=0.0, help="札幌市の陸屋根率(%%)")
    s.add_argument("--flat-other", type=float, default=0.0, help="札幌市以外の陸屋根率(%%)")
    s.add_argument("--roof-note", default="")
    s.add_argument("--pins-work", help="判定済みの作業ディレクトリ(例 work/teine)。家ごとのピンを載せる")
    s.add_argument("--pins-name", default="")
    s.set_defaults(f=cmd_hk_map)
    s = sub.add_parser("roofstats", help="屋根ラベルCSVから陸屋根率"); s.add_argument("labels"); s.set_defaults(f=cmd_roofstats)
    s = sub.add_parser("age-features", help="築年推定: 古い写真(既定2008年度)との比較特徴量")
    s.add_argument("--year", type=int, default=2008); s.add_argument("--procs", type=int, default=4)
    s.set_defaults(f=cmd_age_features)
    s = sub.add_parser("age-review", help="築年推定の目視ラベルページ")
    s.add_argument("--year", type=int, default=2008); s.add_argument("-n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_age_review)
    s = sub.add_parser("roofsample", help="屋根形状ラベル用の標本ページ")
    s.add_argument("-n", type=int, default=150); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_roofsample)
    s = sub.add_parser("aggregate", help="5-6. 集計と GeoJSON/HTML 出力"); s.add_argument("--threshold", type=float)
    s.set_defaults(f=cmd_aggregate)
    s = sub.add_parser("evaluate", help="目視ラベルから精度算出"); s.add_argument("labels")
    s.set_defaults(f=cmd_evaluate)

    a = p.parse_args(argv)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
