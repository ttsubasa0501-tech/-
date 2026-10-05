"""住宅・土地統計調査 令和5年(2023)の市区町村別集計を e-Stat API から取得する。

使う表:
 * 0004021758 所有の関係 × 構造 × 建築の時期(7) × 建て方 × 階数 別住宅数  (市区町村)
 * 0004021627 種類 × 所有の関係 × 建て方 × 構造 × 省エネルギー設備等(太陽光あり/なし) 別住宅数 (市区町村)
 * 0004021645 種類 × 所有の関係 × 建築の時期(9) × 省エネルギー設備等 別住宅数 (市区町村)
アプリケーションID は環境変数 ESTAT_APP_ID から読む（リポジトリには入れない）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.e-stat.go.jp/rest/3.0/app/json"
T_PERIOD = "0004021758"     # 所有 × 建て方 × 建築時期
T_SOLAR_TYPE = "0004021627"  # 所有 × 建て方 × 省エネ設備
T_SOLAR_AGE = "0004021645"   # 所有 × 建築時期 × 省エネ設備


def _app_id() -> str:
    v = os.environ.get("ESTAT_APP_ID")
    if not v:
        raise SystemExit("環境変数 ESTAT_APP_ID が未設定です（e-Stat のアプリケーションID）")
    return v


def hokkaido_areas(stats_id: str = T_PERIOD) -> pd.DataFrame:
    r = requests.get(f"{API}/getMetaInfo", params={"appId": _app_id(), "statsDataId": stats_id}, timeout=60)
    r.raise_for_status()
    objs = r.json()["GET_META_INFO"]["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"]
    area = next(o for o in objs if o["@id"] == "area")["CLASS"]
    df = pd.DataFrame([{"code": a["@code"], "name": a["@name"]} for a in area])
    return df[df["code"].str.startswith("01")].reset_index(drop=True)


def fetch(stats_id: str, areas: list[str], cache: Path, **filters) -> pd.DataFrame:
    """getStatsData を市区町村コードで分割して取得し、列 area, cat*, value の DataFrame にする。"""
    h = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:10]   # 実行間で安定なキー
    key = cache / f"{stats_id}_{h}.csv"
    if key.exists():
        return pd.read_csv(key, dtype=str)
    rows = []
    for i in range(0, len(areas), 40):
        p = {"appId": _app_id(), "statsDataId": stats_id, "cdArea": ",".join(areas[i:i + 40]),
             "limit": 100000, "metaGetFlg": "N", "annotationGetFlg": "N"}
        p.update({f"cd{k.capitalize()}": v for k, v in filters.items()})
        r = requests.get(f"{API}/getStatsData", params=p, timeout=120)
        r.raise_for_status()
        d = r.json()["GET_STATS_DATA"]
        if d["RESULT"]["STATUS"] not in (0, 1):   # 1 = 該当データなし
            raise RuntimeError(d["RESULT"])
        vals = d.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
        vals = vals if isinstance(vals, list) else [vals]
        for v in vals:
            rows.append({k.lstrip("@"): x for k, x in v.items()})
    df = pd.DataFrame(rows).rename(columns={"$": "value"})
    cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(key, index=False)
    return df


def to_number(s: pd.Series) -> pd.Series:
    """'-'(該当なし)は 0、'x'/'…' など秘匿・不詳は NaN。"""
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.replace({"-": "0", "－": "0"}), errors="coerce")


T_STARTS = "0003114504"   # 建築着工統計(年次) 利用関係 × 構造 × 建て方 (市区町村)
START_WEIGHTS = {"2023": 0.25, "2024": 1.0, "2025": 1.0, "2026": 0.75}   # 2023.10〜2026.9 の 3 年分


def fetch_new_builds(work: Path) -> pd.Series:
    """住調(2023.10 時点)以降に建った/建つ一戸建(持家+分譲)の推計戸数を市区町村コードごとに返す。

    建築着工統計の最新年(通常 2024 年)を、データの無い 2025・2026 年にも当てはめる（仮定）。
    """
    cache = work / "estat_cache" / "starts_oneplex.csv"
    if cache.exists():
        d = pd.read_csv(cache, dtype={"area": str, "use": str, "year": str})
    else:
        m = requests.get(f"{API}/getMetaInfo", params={"appId": _app_id(), "statsDataId": T_STARTS}, timeout=60).json()
        objs = m["GET_META_INFO"]["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"]
        area = [a["@code"] for a in next(o for o in objs if o["@id"] == "area")["CLASS"] if a["@code"].startswith("01")]
        rows = []
        for i in range(0, len(area), 50):
            r = requests.get(f"{API}/getStatsData", params={
                "appId": _app_id(), "statsDataId": T_STARTS, "cdArea": ",".join(area[i:i + 50]),
                "cdCat01": "12", "cdCat02": "11", "cdCat03": "12,15", "cdTab": "19",
                "limit": 100000, "metaGetFlg": "N"}, timeout=120).json()["GET_STATS_DATA"]
            v = r.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
            rows += v if isinstance(v, list) else [v]
        d = pd.DataFrame(rows).rename(columns={"@area": "area", "@cat03": "use", "@time": "time", "$": "value"})
        d["year"] = d["time"].str[:4]
        cache.parent.mkdir(parents=True, exist_ok=True)
        d[["area", "use", "year", "value"]].to_csv(cache, index=False)
    d["v"] = to_number(d["value"]).fillna(0)
    by = d.groupby(["area", "year"])["v"].sum().unstack(fill_value=0)
    last = str(max(int(c) for c in by.columns))
    w = START_WEIGHTS
    est = sum(by.get(y, by[last]) * wt for y, wt in w.items())
    return est.rename("new_builds")


# ---------------------------------------------------------------- 係数の組み立て
CITY_TOTAL = "01000"   # 北海道
SAPPORO = "01100"      # 札幌市(区の合計と二重計上になるので除く)
SHRINK_M = 500         # 太陽光設置率を北海道平均へ寄せる強さ(擬似サンプル数)


def _val(df: pd.DataFrame, **cats) -> pd.Series:
    d = df
    for k, v in cats.items():
        d = d[d[k] == v]
    return d.groupby("area")["value"].first() if "v" not in d else d.groupby("area")["v"].first()


def build_factors(work: Path, census_units: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """市区町村ごとの「戸建て → 持ち家 → 築20年以内 → 太陽光なし」の割合を作る。

    census_units: 列 muni(5桁) と detached(2020 国勢調査の一戸建世帯数) を持つ表（市区町村ごと）。
    住調の市区町村別表は市(札幌市は区)のみなので、町村は「北海道 − 市の合計」(町村部)の割合で代用する。
    築20年 = 2006 年以降の築。区分(2001〜2010)を割るため 2006〜2010 の比率を築年×設備表から求める。
    """
    areas = hokkaido_areas()
    A = list(areas["code"])
    cache = work / "estat_cache"
    P = fetch(T_PERIOD, A, cache, cat01="0,1", cat02="0", cat03="1", cat04="00")
    B = fetch(T_SOLAR_AGE, A, cache, cat01="0", cat02="1", cat04="00,21,22")
    C = fetch(T_SOLAR_TYPE, A, cache, cat01="0", cat02="1", cat03="0,1", cat04="0", cat05="00,21,22")
    for d in (P, B, C):
        d["v"] = to_number(d["value"])

    rows = {}
    for code in A:
        g = lambda d, **c: float(d[(d["area"] == code) & pd.concat([d[k] == v for k, v in c.items()], axis=1).all(axis=1)]["v"].sum())
        r = {}
        r["det_all"] = g(P, cat01="0", cat05="00")           # 一戸建(全所有)
        r["det_own"] = g(P, cat01="1", cat05="00")           # 持ち家一戸建
        for p in ("06", "07", "08"):
            r[f"own_p{p}"] = g(P, cat01="1", cat05=p)      # 持ち家一戸建 時期別
        for p in ("07", "08", "09", "10", "11"):           # 持ち家(全建て方) 時期別 総数・太陽光あり
            r[f"b_tot_{p}"] = g(B, cat03=p, cat04="00")
            r[f"b_sol_{p}"] = g(B, cat03=p, cat04="21")
        r["c_all_tot"] = g(C, cat03="0", cat05="00")
        r["c_all_sol"] = g(C, cat03="0", cat05="21")
        r["c_det_tot"] = g(C, cat03="1", cat05="00")
        r["c_det_sol"] = g(C, cat03="1", cat05="21")
        rows[code] = r
    T = pd.DataFrame(rows).T

    cities = [c for c in A if c not in (CITY_TOTAL, SAPPORO)]
    rural = T.loc[CITY_TOTAL] - T.loc[cities].sum()
    rural = rural.clip(lower=0)
    T.loc["rural"] = rural
    hk = T.loc[CITY_TOTAL]

    def derive(r: pd.Series, prior: dict | None) -> dict:
        # 2006〜2010 の割合(2001〜2010 のうち)
        phi_den = r.b_tot_07 + r.b_tot_08
        phi = r.b_tot_08 / phi_den if phi_den >= 100 else (prior or {}).get("phi", 0.5)
        new_own = r.own_p07 + r.own_p08 + phi * r.own_p06
        n_new_all = r.b_tot_08 + r.b_tot_09 + r.b_tot_10 + r.b_tot_11
        sol_new_all = r.b_sol_08 + r.b_sol_09 + r.b_sol_10 + r.b_sol_11
        # 北海道全体の値へ収縮（小さい市の標本誤差を抑える）
        pr = (prior or {}).get("s_new_all", sol_new_all / max(n_new_all, 1))
        s_all = (sol_new_all + SHRINK_M * pr) / (n_new_all + SHRINK_M)
        rate_det = r.c_det_sol / r.c_det_tot if r.c_det_tot >= 300 else None
        rate_all = r.c_all_sol / r.c_all_tot if r.c_all_tot >= 300 else None
        R = (rate_det / rate_all) if (rate_det and rate_all) else (prior or {}).get("R", 1.2)
        R = float(np.clip(R, 0.8, 2.5))
        s_det = float(min(s_all * R, 0.6))
        # 新築(2021〜2023.9 築相当)の太陽光設置率
        n11, sol11 = r.b_tot_11, r.b_sol_11
        pr11 = (prior or {}).get("s11_all", sol11 / max(n11, 1))
        s11_all = (sol11 + SHRINK_M * pr11) / (n11 + SHRINK_M)
        s11_det = float(min(s11_all * R, 0.8))
        own_share = r.det_own / r.det_all if r.det_all else np.nan
        new_share = new_own / r.det_own if r.det_own else np.nan
        return {"phi": phi, "s_new_all": s_all, "s11_all": s11_all, "s_new11_det": s11_det, "R": R, "s_new_det": s_det, "own_share": own_share,
                "new_share": new_share, "n_new_all": n_new_all, "survey_det_all": r.det_all}

    hk_f = derive(hk, None)
    fac = {c: derive(T.loc[c], hk_f) for c in cities + ["rural"]}
    fac["hokkaido"] = hk_f
    F = pd.DataFrame(fac).T
    F["q"] = F["own_share"] * F["new_share"] * (1 - F["s_new_det"])   # 一戸建1軒あたり「築20年以内・持ち家・太陽光なし」の確率
    F["q_before_solar"] = F["own_share"] * F["new_share"]
    return F


def assign_to_census(F: pd.DataFrame, census_units: pd.DataFrame, new_builds: pd.Series | None = None) -> pd.DataFrame:
    """census_units(muni, city_name, detached) の各市区町村に係数を割り当てる（町村は町村部の値）。"""
    out = census_units.copy()
    out["factor_src"] = out["muni"].map(lambda m: m if m in F.index else "rural")
    for c in ("own_share", "new_share", "s_new_det", "s_new11_det", "q", "q_before_solar"):
        out[c] = out["factor_src"].map(F[c])
    out["is_rural_default"] = out["factor_src"] == "rural"
    nb = new_builds if new_builds is not None else pd.Series(dtype=float)
    out["new_builds"] = out["muni"].map(nb).fillna(0.0)          # 2023.10 以降の新築(着工統計からの推計)
    out["nbr"] = np.where(out["detached"] > 0, out["new_builds"] / out["detached"], 0.0)
    out["own_new_old"] = out["detached"] * out["q_before_solar"]  # 住調時点(〜2023.9築)の築20年以内持ち家戸建て
    out["own_new"] = out["own_new_old"] + out["new_builds"]       # 新築を足した築20年以内
    out["pre_roof"] = (out["own_new_old"] * (1 - out["s_new_det"])
                       + out["new_builds"] * (1 - out["s_new11_det"]))   # 太陽光なしまで引いた、陸屋根を引く前の対象戸数
    return out
