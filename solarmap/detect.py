"""太陽光パネル判定器。差し替え可能な Detector インターフェース。

* baseline : 色ヒューリスティック（モデル未導入時の動作確認用。精度は期待しないこと）
* *.pt     : ultralytics YOLO の重み（太陽光パネル検出モデルを別途用意）
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .tiles import Crop


class Detector:
    name = "base"
    threshold = 0.5

    def score(self, crop: Crop) -> float:  # 0..1: パネルらしさ
        raise NotImplementedError


class ColorBaseline(Detector):
    """屋根ポリゴン内で「暗い青灰色」画素の割合を返す簡易判定。"""
    name = "baseline-color"
    threshold = 0.2

    def score(self, crop: Crop) -> float:
        px = crop.img[crop.mask].astype(float)
        if len(px) == 0:
            return 0.0
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        panel_like = (b - r > 10) & (b >= g - 5) & (px.max(axis=1) < 150)
        return float(panel_like.mean())


class YoloDetector(Detector):
    """ultralytics YOLO。建物ポリゴンと十分重なる検出の最大信頼度をスコアとする。"""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 320,
                 class_keywords=("solar", "panel", "pv"), min_overlap: float = 0.3):
        from ultralytics import YOLO  # 任意依存
        self.model = YOLO(weights)
        self.name = f"yolo:{weights}"
        self.conf, self.imgsz, self.min_overlap = conf, imgsz, min_overlap
        names = self.model.names
        sel = [i for i, n in names.items() if any(k in str(n).lower() for k in class_keywords)]
        self.classes = sel or None  # 一致なしなら全クラスを使う
        self.threshold = conf

    def score(self, crop: Crop) -> float:
        h, w = crop.img.shape[:2]
        scale = max(1.0, self.imgsz / max(h, w))
        im = Image.fromarray(crop.img).resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        res = self.model.predict(np.asarray(im), conf=self.conf, imgsz=self.imgsz,
                                 classes=self.classes, verbose=False)[0]
        best = 0.0
        for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = (box / scale).round().astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            area = max((x1 - x0) * (y1 - y0), 1)
            if crop.mask[y0:y1, x0:x1].sum() / area >= self.min_overlap:
                best = max(best, float(c))
        return best


def make_detector(spec: str, **kw) -> Detector:
    if spec == "baseline":
        return ColorBaseline()
    if spec.endswith(".pt"):
        return YoloDetector(spec, **kw)
    raise ValueError(f"未知の検出器: {spec}（'baseline' か YOLO の .pt パス）")
