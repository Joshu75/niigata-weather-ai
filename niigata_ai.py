import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

print("AIモデル構築スタート！データを取得・準備しています...")

# --- 1. データの取得（過去40日分の実測データ） ---
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=40)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.9161,
    "longitude": 139.0364,
    "start_date": start_date.strftime('%Y-%m-%d'),
    "end_date": end_date.strftime('%Y-%m-%d'),
    "hourly": "temperature_2m",
    "timezone": "Asia/Tokyo"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame({
    "date": pd.to_datetime(data["hourly"]["time"]),
    "temp": data["hourly"]["temperature_2m"]
})
df.dropna(inplace=True)

# --- 2. AIに学習させるための「特徴量」を作成（ここが職人技！） ---
# 気温の変化には「時間帯」や「前日の気温」が大きく影響します
df['hour'] = df['date'].dt.hour
df['day_of_year'] = df['date'].dt.dayofyear
# 24時間前（昨日）の同じ時間の気温を「明日の予測」のヒントにする
df['temp_24h_ago'] = df['temp'].shift(24) 

# shift(24)によって最初の24時間分は「昨日のデータがない（NaN）」状態になるため削除
df.dropna(inplace=True)

# --- 3. データの分割（学習用とテスト用） ---
# 最初の35日間をAIの「学習用(Train)」、直近5日間を「答え合わせ用(Test)」に分けます
train_size = int(len(df) * (35/40))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# AIへの入力(X)と、当てさせたい正解(y)に分ける
features = ['hour', 'day_of_year', 'temp_24h_ago']
X_train, y_train = train_df[features], train_df['temp']
X_test, y_test = test_df[features], test_df['temp']

# --- 4. AIモデルの学習 ---
print("AIが過去の気温パターンを学習中...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # ここで学習が完了！

# --- 5. 学習したAIで「答え合わせ用データ」を予測してみる ---
# AIに「直近5日間の時間帯と前日気温」だけを渡し、自力で気温を予測させます
predictions = model.predict(X_test)

# --- 6. 結果のグラフ化 ---
plt.figure(figsize=(12, 6))

# 学習に使った過去のデータ（直近3日分だけ表示）
plt.plot(train_df['date'].tail(72), train_df['temp'].tail(72), color='gray', linestyle='--', label='Past Data (Train)')

# 答え合わせ：実際の気温（緑） vs AIの予測（オレンジ）
plt.plot(test_df['date'], y_test, color='#2CA02C', label='Actual Temp (Ground Truth)', linewidth=2)
plt.plot(test_df['date'], predictions, color='#FF5733', label='AI Prediction', linestyle='-', linewidth=2)

plt.title('AI Weather Prediction vs Actual Temperature (Niigata City)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (℃)', fontsize=12)
plt.axvline(x=test_df['date'].iloc[0], color='blue', linestyle=':', label='AI Start Predicting') # 予測開始ライン
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()
print("完了！グラフでAIの予測精度を確認してください。")