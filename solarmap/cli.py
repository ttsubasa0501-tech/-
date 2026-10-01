"""CLI。各ステップは work ディレクトリ内の成果物でつながる（途中再開可）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import aggregate as agg
from . import buildings as bl
from . import estat, fgd, mapout, review, vtile
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
    rows = []
    for i, r in enumerate(b.itertuples()):
        c = building_crop(r.geometry, store)
        ok = c.missing == 0
        s = det.score(c) if ok else 0.0
        rows.append({"bid": r.bid, "score": s, "pred": bool(ok and s >= det.threshold), "valid": ok})
        if (i + 1) % 2000 == 0:
            print(f"[detect] {i+1}/{len(b)}")
    df = pd.DataFrame(rows)
    df.to_csv(w / "predictions.csv", index=False)
    (w / "detector.json").write_text(json.dumps({"name": det.name, "threshold": det.threshold}))
    print(f"[detect] {det.name}: 有効 {df.valid.sum()}/{len(df)}, パネルあり {df.pred.sum()}")


def cmd_aggregate(a):
    w = Path(a.work)
    boundary = gpd.read_file(w / "boundary.gpkg")
    b = gpd.read_file(w / "buildings.gpkg")
    pred = pd.read_csv(w / "predictions.csv")
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
    s = sub.add_parser("aggregate", help="5-6. 集計と GeoJSON/HTML 出力"); s.set_defaults(f=cmd_aggregate)
    s = sub.add_parser("evaluate", help="目視ラベルから精度算出"); s.add_argument("labels")
    s.set_defaults(f=cmd_evaluate)

    a = p.parse_args(argv)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
