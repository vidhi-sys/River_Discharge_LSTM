import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# Load Data
# ==============================
df = pd.read_csv("riverdischarge_manual_daily_madhya-pradesh-sw_mp_1950_2000.csv")

# Convert datetime
df["Data Acquisition Time"] = pd.to_datetime(
    df["Data Acquisition Time"],
    dayfirst=True,
    errors="coerce"
)
df = df.dropna(subset=["Data Acquisition Time"])
df = df.sort_values("Data Acquisition Time")

# ==============================
# Feature Engineering
# ==============================
df["month"] = df["Data Acquisition Time"].dt.month
df["day"] = df["Data Acquisition Time"].dt.day
df["year"] = df["Data Acquisition Time"].dt.year

# Lag feature (previous day discharge)
df["lag_1"] = df["Manual Daily River Water Discharge (m3/sec)"].shift(1)

df = df.dropna()

target = "Manual Daily River Water Discharge (m3/sec)"
features = ["month", "day", "year", "lag_1"]

X = df[features]
y = df[target]

# ==============================
# RANDOM FOREST
# ==============================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

joblib.dump(rf, "rf_model.pkl")

# ==============================
# LSTM MODEL
# ==============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[[target]])

joblib.dump(scaler, "scaler.pkl")

# Create sequences
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_data)

model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=5)

model.fit(X_lstm, y_lstm, epochs=50, batch_size=16,
          validation_split=0.2, callbacks=[early_stop])

model.save("lstm_model.h5")

print("✅ Models Saved Successfully!")
