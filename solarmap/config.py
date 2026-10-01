"""定数。札幌市の行政区コードなど。"""

# 全国地方公共団体コード（札幌市の各区）
WARDS = {
    "中央区": "01101", "北区": "01102", "東区": "01103", "白石区": "01104",
    "厚別区": "01105", "豊平区": "01106", "清田区": "01107", "南区": "01108",
    "西区": "01109", "手稲区": "01110",
}

# 札幌市は平面直角座標系 XII 系（JGD2011）。面積計算に使う。
CRS_AREA = "EPSG:6680"
CRS_GEO = "EPSG:6668"  # JGD2011 緯度経度

# 国土地理院シームレス空中写真（z18 が最大。北海道で約 0.44 m/px）
TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
TILE_ZOOM = 18
USER_AGENT = "sapporo-solar-map-pilot/0.1 (research; contact: see repository)"

# e-Stat 地図で見る統計 (統計GIS) のダウンロード URL（2020 年国勢調査 小地域）
ESTAT_BOUNDARY_URL = (
    "https://www.e-stat.go.jp/gis/statmap-search/data"
    "?dlserveyId=A002005212020&code={code}&coordSys=1&format=shape&downloadType=5"
)
ESTAT_STATS_URL = (
    "https://www.e-stat.go.jp/gis/statmap-search/data"
    "?statsId={stats_id}&code={code}&downloadType=2"
)
