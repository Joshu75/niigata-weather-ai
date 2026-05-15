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
    "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m",
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
    "wind_speed": res_fore["hourly"]["wind_speed_10m"]
}).dropna()

df_future['hour'] = df_future['date'].dt.hour
df_future['day_of_year'] = df_future['date'].dt.dayofyear


print("【ステップ4】LightGBMが新潟市の未来を独自予測中！")

# --- 4. LightGBMによる未来の予測 ---
X_future = df_future[features]
df_future['ai_temp'] = model.predict(X_future)


# --- 5. 結果のグラフ化 ---
plt.figure(figsize=(12, 6))

# 気象庁の予測（グレーの点線）
plt.plot(df_future['date'], df_future['jma_temp'], color='gray', linestyle='--', label='JMA MSM Raw Forecast', linewidth=2)
# LightGBMの独自予測（青い線）
plt.plot(df_future['date'], df_future['ai_temp'], color='#3498DB', label='LightGBM Corrected Forecast', linewidth=2)

plt.title('Advanced Weather Forecast: JMA Raw vs LightGBM (Niigata City)', fontsize=14)
plt.xlabel('Date / Time', fontsize=12)
plt.ylabel('Temperature (℃)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# 画面に出すのではなく、画像ファイルとして保存する！
plt.savefig('niigata_forecast.png', dpi=300)
print("予測完了！グラフを 'niigata_forecast.png' として保存しました。")
