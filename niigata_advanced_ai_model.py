import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

print("進化したAIモデルを構築中！複数の気象データを取得しています...")

# --- 1. データの取得（過去40日分） ---
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=40)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.9161,
    "longitude": 139.0364,
    "start_date": start_date.strftime('%Y-%m-%d'),
    "end_date": end_date.strftime('%Y-%m-%d'),
    # ★ここが進化ポイント！気温に加えて「湿度」「日射量」「風速」を取得します
    "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m",
    "timezone": "Asia/Tokyo"
}

response = requests.get(url, params=params)
data = response.json()

# DataFrameの作成
df = pd.DataFrame({
    "date": pd.to_datetime(data["hourly"]["time"]),
    "temp": data["hourly"]["temperature_2m"],
    "humidity": data["hourly"]["relative_humidity_2m"],
    "radiation": data["hourly"]["shortwave_radiation"], # 日射量（超重要）
    "wind_speed": data["hourly"]["wind_speed_10m"]      # 風速
})
df.dropna(inplace=True)

# --- 2. 特徴量の作成 ---
df['hour'] = df['date'].dt.hour
df['day_of_year'] = df['date'].dt.dayofyear

# --- 3. データの分割（学習用35日、答え合わせ用5日） ---
train_size = int(len(df) * (35/40))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# ★AIに与えるヒント（特徴量）を大幅に増やします
features = ['hour', 'day_of_year', 'humidity', 'radiation', 'wind_speed']

X_train, y_train = train_df[features], train_df['temp']
X_test, y_test = test_df[features], test_df['temp']

# --- 4. AIモデルの学習 ---
print("AIが「日差し」や「風」と「気温」の関係を学習中...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. 学習したAIで予測 ---
predictions = model.predict(X_test)

# --- 6. 結果のグラフ化 ---
plt.figure(figsize=(12, 6))

# 答え合わせ：実際の気温（緑） vs 進化したAIの予測（赤）
plt.plot(test_df['date'], y_test, color='#2CA02C', label='Actual Temp (Ground Truth)', linewidth=2)
plt.plot(test_df['date'], predictions, color='#E74C3C', label='Advanced AI Prediction', linestyle='-', linewidth=2)

plt.title('Advanced AI Weather Prediction (with Solar Radiation & Wind)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (℃)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()
print("完了！日射量を知ったAIの精度を確認してください。")