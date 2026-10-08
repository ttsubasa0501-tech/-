"""全棟の屋根画像(現在・過去)を学習済み視覚モデル(DINOv2 small)で特徴ベクトルにする。

ラベル(築年・屋根の形・太陽光)が届いたら、この特徴ベクトルでロジスティック回帰等を学習する。
途中保存して再開できる。出力: <work>/embed_{now,old}.npy (N x 768, float16) と embed_done.npy
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from .age import OLD_URL
from .tiles import TileStore, building_crop

SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _prep(img: np.ndarray) -> np.ndarray:
    im = Image.fromarray(img).resize((SIZE, SIZE), Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1)


def run(work: Path, year: int = 2008, batch: int = 64, margin: int = 10, threads: int = 3, every: int = 20) -> None:
    work = Path(work)
    torch.set_num_threads(threads)
    b = gpd.read_file(work / "buildings.gpkg")
    n = len(b)
    new = TileStore(work / "tiles", offline=True)
    old = TileStore(work / f"tiles_{year}", url=OLD_URL.replace("{year}", str(year)), offline=True, ext="png")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    f_now, f_old, f_done = work / "embed_now.npy", work / "embed_old.npy", work / "embed_done.npy"
    if f_done.exists():
        E_now, E_old, done = np.load(f_now), np.load(f_old), np.load(f_done)
        print(f"[embed] 再開: {int(done.sum())}/{n}", flush=True)
    else:
        E_now, E_old = np.zeros((n, 768), np.float16), np.zeros((n, 768), np.float16)
        done = np.zeros(n, bool)

    def emb(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = model(pixel_values=torch.from_numpy(x)).last_hidden_state
        return torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy().astype(np.float16)

    todo = np.where(~done)[0]
    for k, s in enumerate(range(0, len(todo), batch)):
        idx = todo[s:s + batch]
        xn, xo = [], []
        for i in idx:
            g = b.geometry.iloc[i]
            try:
                xn.append(_prep(building_crop(g, new, margin=margin).img))
                xo.append(_prep(building_crop(g, old, margin=margin).img))
            except Exception:
                xn.append(np.zeros((3, SIZE, SIZE), np.float32))
                xo.append(np.zeros((3, SIZE, SIZE), np.float32))
        E_now[idx] = emb(np.stack(xn))
        E_old[idx] = emb(np.stack(xo))
        done[idx] = True
        if (k + 1) % every == 0 or s + batch >= len(todo):
            np.save(f_now, E_now); np.save(f_old, E_old); np.save(f_done, done)
            print(f"[embed] {int(done.sum())}/{n}", flush=True)
    np.save(f_now, E_now); np.save(f_old, E_old); np.save(f_done, done)
    (work / "embed_finished.txt").write_text("ok")
