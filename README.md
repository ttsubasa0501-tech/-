# 札幌市 町丁目別 住宅用太陽光パネル設置率マップ（手稲区パイロット）

町丁目ごとに **設置率 = パネルありと判定した戸建て候補の棟数 / 国勢調査の一戸建世帯数** を算出し、
GeoJSON と folium の HTML 地図を出力します。検出精度を **100 件目視確認できるページ**も生成します。

## 実行手順（手稲区 = 01109）

```bash
pip install -r requirements.txt mapbox-vector-tile
# 検出器に YOLO を使う場合: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install ultralytics
W="--work work/teine"

# 1. e-Stat（どちらも自動ダウンロード）
python -m solarmap $W boundary                 # 小地域境界 (366 町丁目)
python -m solarmap $W households               # 一戸建世帯数 (2020 国勢調査 住宅の建て方 T001086)

# 2. 建物外周線 → 50〜250㎡・コンパクト形状の戸建て候補
python -m solarmap $W buildings --types 普通建物 堅ろう建物                 # 国土地理院ベクトルタイル(BldA)から自動取得
python -m solarmap $W buildings data/fgd/*.zip --types 普通建物 堅ろう建物  # 基盤地図情報 GML を使う場合

# 3. 国土地理院 空中写真タイル (z18) をキャッシュ
python -m solarmap $W tiles --workers 8

# 4. パネル判定（YOLO 重みは下記）
python -m solarmap $W detect --detector work/models/finloop.pt --conf 0.1 --imgsz 480 --offline

# 目視確認: work/teine/review/review.html をブラウザで開く
python -m solarmap $W review -n 100 --offline
python -m solarmap $W evaluate review_labels.csv            # ページからCSV保存したもの（任意）

# 5-6. 集計・出力 → work/teine/solar_rate.geojson, solar_rate_map.html
python -m solarmap $W aggregate
```
`--ward 北区` 等で他区にも使えます（`solarmap/config.py` の WARDS。区コードは手稲区 01109、清田区 01110）。
## 目視確認（100 件）
- 「パネルあり判定」50 件 + 「なし判定」50 件の**層化抽出**（設置率が数%だとランダム 100 件では
  パネルありが数件しか入らず評価できないため）。
- 各カードは 元画像 / 建物外周線つき の並置、地理院地図へのリンク付き。ラジオボタンで あり/なし/判別不能。
- ページ上部に **適合率・見逃し率・推定再現率**をリアルタイム表示（層の母数で重み付け）。入力は localStorage に保存、CSV 出力可。

## 実データでの検証結果（手稲区）
- 境界: 366 町丁目。面積合計 = 和集合（重複なし）。
- 一戸建世帯数: T001086。境界に結合した合計 33,896 = 区合計と一致。秘匿(X) 4 地区は欠損、`-` は 0。
- 建物: 国土地理院ベクトルタイル 77,982 棟 → 戸建て候補 38,678 棟。町丁目別の候補数と一戸建世帯数の相関 0.95、比の中央値 1.1。
- 空中写真タイル 2,156 枚（35MB）を取得。

## 検出モデルについて（重要）
Hugging Face の MIT ライセンス YOLO 重みを 2 種試した。ランダム 400 棟での反応は
`Cyrille37/solar-panels-IGN-bdortho` がほぼ 0 件、`finloop/yolov8s-seg-solar-panels` が 36 件(conf 0.03)。
後者(`--conf 0.1 --imgsz 480`)で手稲区 38,678 棟を判定すると 1,194 棟(3.1%)がパネルあり判定、全体の設置率は 3.5%(一戸建世帯数ベース)。
スコア上位 24 棟を私が見た範囲では、格子状のパネルアレイが明瞭なものが大半だが、平屋根の事業用建物も混じる。
スコアが低い側(0.1〜0.2、判定の半数)の確からしさは未確認。**適合率・再現率は 100 件の目視確認(`results/teine/review.html`)で測る。**
`baseline`(色判定)は青灰色屋根を全て拾うため手稲区では使えない。

## 成果物（手稲区パイロット, `results/teine/`）
- `solar_rate.geojson` … 町丁目別（`rate` と色 `fill` 付き）
- `solar_rate_map.html` … folium 地図（町丁目の塗り分け、空中写真レイヤ、パネルあり判定の建物点）
- `review.html` … 目視確認ページ（画像埋め込みの単一 HTML）と `review_sample.csv`

### 既知の限界
- 面積は**建築面積**（外周線）で代用（床面積ではない）。2 階建ては床面積が約 2 倍になるため `--min-area/--max-area` で調整。
- 分子（候補建物のパネル）と分母（一戸建世帯数）は対象が完全一致しないため、設置率は目安（1 を超えうる）。
- 空中写真の撮影年が地域により異なり、積雪・影・撮影時期の影響を受ける。
- 国土地理院タイルは出典明記が必要（地図の attribution に設定済み）。大量取得は `--delay/--workers` で負荷を抑える。
- ベクトルタイルはタイル境界で切れた建物断片を除外している（重心がタイル本体内の建物のみ採用）。
