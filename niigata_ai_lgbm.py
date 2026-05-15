import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import lightgbm as lgb
import warnings

# LightGBMの不要な警告を非表示にする
warnings.filterwarnings('ignore')

print("【ステップ1】過去のデータ(40日分)から新潟市の気象パターンを学習中...")

# --- 1. 過去データの取得 ---
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=40)

url_archive = "https://archive-api.open-meteo.com/v1/archive"
params_archive = {
    "latitude": 37.9161,
    "longitude": 139.0364,
    "start_date": start_date.strftime('%Y-%m-%d'),
    "end_date": end_date.strftime('%Y-%m-%d'),
    "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m",
    "timezone": "Asia/Tokyo"
}
res_arch = requests.get(url_archive, params=params_archive).json()
df_train = pd.DataFrame({
    "date": pd.to_datetime(res_arch["hourly"]["time"]),
    "temp": res_arch["hourly"]["temperature_2m"],
    "humidity": res_arch["hourly"]["relative_humidity_2m"],
    "radiation": res_arch["hourly"]["shortwave_radiation"],
    "wind_speed": res_arch["hourly"]["wind_speed_10m"]
}).dropna()

# 特徴量の追加
df_train['hour'] = df_train['date'].dt.hour
df_train['day_of_year'] = df_train['date'].dt.dayofyear

features = ['hour', 'day_of_year', 'humidity', 'radiation', 'wind_speed']
X_train, y_train = df_train[features], df_train['temp']


# --- 2. LightGBMモデルの構築と学習 ---
print("【ステップ2】最新のアルゴリズム「LightGBM」で学習中...")
# LightGBM用のデータセットを作成
lgb_train = lgb.Dataset(X_train, y_train)

# パラメータの設定（今回は回帰問題なので regression を指定）
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': 42,
    'verbose': -1 # 学習中のログを非表示
}

# 学習の実行
model = lgb.train(params, lgb_train, num_boost_round=100)


print("【ステップ3】気象庁(JMA MSM)の未来予測データを取得中...")

# --- 3. 気象庁の「未来の予測データ」を取得 ---
url_forecast = "https://api.open-meteo.com/v1/forecast"
params_forecast = {
    "latitude": 37.9161,
    "longitude": 139.0364,
    # ▼ ここに precipitation と wind_direction_10m を足します
    "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m,precipitation,wind_direction_10m",
    "models": "jma_msm",
    "timezone": "Asia/Tokyo",
    "forecast_days": 3
}
res_fore = requests.get(url_forecast, params=params_forecast).json()
df_future = pd.DataFrame({
    "date": pd.to_datetime(res_fore["hourly"]["time"]),
    "jma_temp": res_fore["hourly"]["temperature_2m"],
    "humidity": res_fore["hourly"]["relative_humidity_2m"],
    "radiation": res_fore["hourly"]["shortwave_radiation"],
    "wind_speed": res_fore["hourly"]["wind_speed_10m"],
    # ▼ この2行を追加します（上の行の終わりにカンマが必要です）
    "precipitation": res_fore["hourly"]["precipitation"],
    "wind_dir": res_fore["hourly"]["wind_direction_10m"]
}).dropna()

df_future['hour'] = df_future['date'].dt.hour
df_future['day_of_year'] = df_future['date'].dt.dayofyear


print("【ステップ4】LightGBMが新潟市の未来を独自予測中！")

# --- 4. LightGBMによる未来の予測 ---
X_future = df_future[features]
df_future['ai_temp'] = model.predict(X_future)


# --- 5. 結果のグラフ化 ---
import numpy as np # 風向きの計算（サイン・コサイン）に使うため追加します

# === グラフの描画（プロ仕様：2段メテオグラム） ===
# 画面を縦に2つ分割します（少し高さを抑えて見やすくします）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --------------------------------------------------
# --- 1段目：気温（左Y軸） ＋ 降水量（右Y軸①） ＋ 湿度（右Y軸②） ---
# --------------------------------------------------

# ① 左側のY軸（気温：折れ線グラフ）
ax1.plot(df_future['date'], df_future['jma_temp'], label='JMA Temp (Raw)', color='gray', linestyle='--')
# ※ predictions はご自身の変数名に合わせてください
ax1.plot(df_future['date'], df_future['ai_temp'], label='AI Temp (Corrected)', color='#e74c3c', linewidth=2)
ax1.set_title("Advanced Meteogram: Temp, Precip, Humidity & Wind (Niigata City)")
ax1.set_ylabel("Temperature (°C)")
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# ② 右側のY軸その1（降水量：棒グラフ）
ax1_precip = ax1.twinx() # 右側にもう一つのY軸を作る必殺技
ax1_precip.bar(df_future['date'], df_future['precipitation'], color='#3498db', alpha=0.4, width=0.05, label='Precipitation')
ax1_precip.set_ylabel("Precipitation (mm)", color='#2980b9')
ax1_precip.set_ylim(0, max(df_future['precipitation'].max() + 5, 20)) # 雨が見やすいように上限を調整

# ③ 右側のY軸その2（湿度：点線グラフ）
ax1_humid = ax1.twinx()
ax1_humid.spines['right'].set_position(('axes', 1.08)) # 目盛りが重ならないように枠線の外側に押し出す
ax1_humid.plot(df_future['date'], df_future['humidity'], color='#2ecc71', linestyle=':', linewidth=1.5, label='Humidity')
ax1_humid.set_ylabel("Humidity (%)", color='#27ae60')
ax1_humid.set_ylim(0, 100) # 湿度は0〜100%

# --------------------------------------------------
# --- 2段目：風速（折れ線） ＋ 風向（矢印ベクトル） ---
# --------------------------------------------------

# ① 風速（折れ線グラフ）
ax2.plot(df_future['date'], df_future['wind_speed'], color='#f39c12', linewidth=2, label='Wind Speed')
ax2.set_ylabel("Wind Speed (m/s)")
ax2.set_xlabel("Date / Time")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(0, max(df_future['wind_speed'].max() + 5, 15)) # 矢印を描くための空間を上に作る

# ② 風向（矢印の描画）
# 風向（角度）を数学のラジアンに変換し、矢印のタテヨコの長さを計算
u = -np.sin(np.radians(df_future['wind_dir']))
v = -np.cos(np.radians(df_future['wind_dir']))
# 全部描画すると矢印で真っ黒になるため、[::3]を使って「3時間ごと」に間引きして矢印を配置
ax2.quiver(df_future['date'][::3], df_future['wind_speed'][::3], u[::3], v[::3], 
           color='#d35400', scale=25, width=0.003, headwidth=4, label='Wind Direction')

# グラフ全体の保存設定
plt.xticks(rotation=45)
# 押し出した湿度の目盛りが画像から見切れないように、自動で余白を調整して保存
plt.savefig('niigata_forecast.png', dpi=300, bbox_inches='tight') 
print("プロ仕様の2段メテオグラム画像を保存しました！")