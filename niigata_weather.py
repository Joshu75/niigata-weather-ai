import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("AIの教師データとなる「過去の実測値」を取得中...")

# --- 1. 取得期間の設定（過去1ヶ月分） ---
# 今日から30日前〜昨日までのデータを取得します
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=30)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# --- 2. Open-Meteo Historical APIの設定 ---
lat = 37.9161
lon = 139.0364
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_str,
    "end_date": end_str,
    "hourly": "temperature_2m",  # 実測の気温
    "timezone": "Asia/Tokyo"
}

# --- 3. データの取得とパース ---
try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    print(f"API通信エラー: {e}")
    exit()

# DataFrameに変換
hourly = data["hourly"]
df_actual = pd.DataFrame({
    "date": pd.to_datetime(hourly["time"]),
    "actual_temp": hourly["temperature_2m"]
})

# --- 4. グラフ化して確認 ---
plt.figure(figsize=(12, 6))
plt.plot(df_actual["date"], df_actual["actual_temp"], color='#2CA02C', label='Actual Temperature (℃)', linewidth=1.5)

plt.title(f'Niigata City - Historical Temperature ({start_str} to {end_str})', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (℃)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

print("\nデータの先頭5行（これをAIに学習させます）:")
print(df_actual.head())