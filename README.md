# 札幌市 町丁目別 住宅用太陽光パネル設置率マップ（手稲区パイロット）

町丁目ごとに **設置率 = パネルありと判定した戸建て候補の棟数 / 国勢調査の一戸建世帯数** を算出し、
GeoJSON と folium の HTML 地図を出力します。検出精度を **100 件目視確認できるページ**も生成します。

## 実行手順（手稲区 = 01110）

```bash
pip install -r requirements.txt
W="--work work/teine"

# 1. e-Stat
python -m solarmap $W boundary                      # 小地域境界 (自動DL。失敗時は --file で zip 展開済み .shp を指定)
python -m solarmap $W households --file stats.csv   # 一戸建世帯数 (下記「要確認」参照)

# 2. 基盤地図情報 BldA（要ログインDL。手稲区を覆う FG-GML の .zip/.xml を data/fgd/ に置く）
python -m solarmap $W buildings data/fgd/*.zip      # 50〜250㎡・コンパクト形状で戸建て候補抽出

# 3. 国土地理院 空中写真タイル (z18) をキャッシュ
python -m solarmap $W tiles

# 4. パネル判定
python -m solarmap $W detect --detector baseline            # 動作確認用の色ヒューリスティック
python -m solarmap $W detect --detector path/to/solar.pt    # YOLO 重み (pip install ultralytics)

# 目視確認: work/teine/review/review.html をブラウザで開く
python -m solarmap $W review -n 100
python -m solarmap $W evaluate review_labels.csv            # ページからCSV保存したもの（任意）

# 5-6. 集計・出力 → work/teine/solar_rate.geojson, solar_rate_map.html
python -m solarmap $W aggregate
```
`--ward 北区` 等で他区にも使えます（`solarmap/config.py` の WARDS）。

## 目視確認（100 件）
- 「パネルあり判定」50 件 + 「なし判定」50 件の**層化抽出**（設置率が数%だとランダム 100 件では
  パネルありが数件しか入らず評価できないため）。
- 各カードは 元画像 / 建物外周線つき の並置、地理院地図へのリンク付き。ラジオボタンで あり/なし/判別不能。
- ページ上部に **適合率・見逃し率・推定再現率**をリアルタイム表示（層の母数で重み付け）。入力は localStorage に保存、CSV 出力可。

## 検証状況（正直な報告）
開発セッションの egress ポリシーで **e-Stat・国土地理院・HuggingFace に接続できなかった**ため、
実データでの実行は未検証です。検証したのは合成データ（境界・FGD 風 GML・空中写真タイル・世帯数 CSV）に対する
エンドツーエンド試験のみです（`pytest`、3 件パス）。

### 実データで初回に確認が必要な点
1. **一戸建世帯数**: 2020 年国勢調査 小地域の「住宅の建て方」表の統計ID(`--stats-id`)は未確認です。
   e-Stat「地図で見る統計」で手稲区・住宅の建て方を選び CSV を保存して `--file` で渡すのが確実です。
   列名に「一戸建」を含む列を自動選択します（複数あれば警告）。
2. **境界DL URL** (`config.py` の `ESTAT_BOUNDARY_URL`) は既知のパターンに基づく未検証の値です。
3. **基盤地図情報 BldA** はダウンロードにログインが必要なため自動取得していません。
   GML は `Surface/PolygonPatch` と `Polygon` の両形式、複数 `posList` 連結に対応（合成データで確認）。
4. **検出モデル**: 「オープンソースの太陽光パネル検出モデル」は特定できていません（HF 等に接続できず確認不可）。
   YOLO 形式 `.pt` を `--detector` に渡せます。`baseline` は色だけの簡易判定で、**精度評価の対象外の動作確認用**です。
   z18 は北海道で約 0.44 m/px と粗く、高解像度画像で学習したモデルは精度が落ちる可能性があります → まさに 100 件目視で測る点です。

### 既知の限界
- 面積は**建築面積**（外周線）で代用（床面積ではない）。2 階建ては床面積が約 2 倍になるため `--min-area/--max-area` で調整。
- 分子（候補建物のパネル）と分母（一戸建世帯数）は対象が完全一致しないため、設置率は目安（1 を超えうる）。
- 空中写真の撮影年が地域により異なり、積雪・影・撮影時期の影響を受ける。
- 国土地理院タイルは出典明記が必要（地図の attribution に設定済み）。大量取得は `--delay/--workers` で負荷を抑える。
