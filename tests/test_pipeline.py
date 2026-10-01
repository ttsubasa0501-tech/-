import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from solarmap import fgd, review
from solarmap.cli import main
from solarmap.tiles import lonlat_to_pixel, meters_per_pixel
from tests import synthetic


def test_pixel_math():
    # スリッピーマップ標準式(tan/sec 形)で独立に算出したタイル番号と一致すること
    import math
    lon, lat, z = 139.7671, 35.6812, 18
    tx = int((lon + 180) / 360 * 2 ** z)
    ty = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z)
    x, y = lonlat_to_pixel(lon, lat, z)
    assert (int(x // 256), int(y // 256)) == (tx, ty)
    assert meters_per_pixel(43.1, 18) == pytest.approx(0.436, abs=0.005)


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    return root, synthetic.make(root)


def test_fgd_parse_multi_segment(syn):
    _, s = syn
    g = fgd.load_bldA([s["gml"]])
    assert len(g) == s["n_total"]
    assert g.geometry.is_valid.all()
    assert abs(g.geometry.iloc[0].centroid.y - 43.12) < 0.02   # 緯度経度の順序が正しい


def test_e2e(syn):
    root, s = syn
    w = str(root / "work")
    base = ["--work", w]
    main(base + ["boundary", "--file", str(s["boundary"])])
    main(base + ["households", "--file", str(s["csv"])])
    main(base + ["buildings", str(s["gml"])])
    # タイルは合成データを work/tiles にコピーして offline 実行
    import shutil
    shutil.copytree(root / "tiles", Path(w) / "tiles")
    main(base + ["tiles", "--offline"])
    main(base + ["detect", "--offline"])
    main(base + ["aggregate"])
    main(base + ["review", "--offline", "-n", "100"])

    b = gpd.read_file(Path(w) / "buildings.gpkg")
    assert len(b) == sum(s["n_house"].values())          # 物置(20㎡)・大型(500㎡)は除外
    out = gpd.read_file(Path(w) / "solar_rate.geojson").set_index("S_NAME")
    for k in "ABCD":
        row = out.loc[f"{k}町丁目"]
        assert row.n_candidates == s["n_house"][k]
        assert row.n_panel == s["n_panel"][k]            # 合成画像では完全に検出できる
        assert row.rate == pytest.approx(s["n_panel"][k] / synthetic.DETACHED[k])
        assert row.fill.startswith("#")
    assert (Path(w) / "solar_rate_map.html").stat().st_size > 10_000
    page = Path(w) / "review" / "review.html"
    assert page.exists() and len(list((Path(w) / "review" / "img").glob("*.jpg"))) == 100

    # 目視ラベルの評価（サンプルの予測をそのまま正解にすれば precision/recall = 1）
    sm = pd.read_csv(Path(w) / "review" / "review_sample.csv")
    lab = pd.DataFrame({"bid": sm.bid, "pred": sm.pred, "truth": sm.pred.map({True: "yes", False: "no"})})
    pred = pd.read_csv(Path(w) / "predictions.csv")
    res = review.evaluate(lab, {"pos": int(pred.pred.sum()), "neg": int((~pred.pred).sum())})
    assert res["precision"] == 1 and res["est_recall"] == 1
