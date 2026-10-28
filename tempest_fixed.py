"""TempestAI with a leak-free evaluation.

Same data (Open-Meteo archive, Ahmedabad, 2024-02-11 to 2025-04-05), same features and the same
four architectures as "Temp-Pred (Extensive DL).ipynb". What changes:

1. Chronological split. The series is cut into contiguous train / validation / test blocks
   (70 / 15 / 15 %), with a 48-hour gap between blocks so no window straddles two of them. The
   notebook shuffled overlapping windows with train_test_split, so test windows had near-copies
   in training.
2. Real forecasting. Inputs are the 24 hours *before* time t (including past temperature);
   targets are the 24 hours *from* t. In the notebook the inputs covered hours i..i+24 and the
   targets i+1..i+25, so the model mostly read temperature off same-time weather.
3. Scaler fitted on the training block only; early stopping on validation, never on test.
4. A persistence baseline ("tomorrow repeats the last 24 hours") so the scores have a reference.
5. For comparison, the notebook's original evaluation is reproduced for GRU_Attention.

Writes results.json next to --out.
"""
import argparse
import json
import os
import random

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (GRU, LSTM, Attention, BatchNormalization, Conv1D, Dense, Dropout,
                                     Flatten, Input, Lambda, RepeatVector, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

H = 24  # hours of history in, hours of forecast out
SEED = 42
MODELS = ["GRU_Attention", "LSTM", "CNN_GRU", "Dense"]


def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def fetch():
    """Identical to the notebook's fetch_and_preprocess()."""
    client = openmeteo_requests.Client(session=requests_cache.CachedSession(".cache", expire_after=-1))
    params = {
        "latitude": 23.0258, "longitude": 72.5873, "start_date": "2024-02-11", "end_date": "2025-04-05",
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
                   "wind_speed_10m", "cloud_cover", "surface_pressure"],
    }
    hourly = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0].Hourly()
    index = pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                          end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                          freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left")
    df = pd.DataFrame({name: hourly.Variables(i).ValuesAsNumpy() for i, name in enumerate(params["hourly"])},
                      index=index).dropna()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df["weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)
    return df


def build_model(model_type, input_shape):
    """The notebook's four architectures, unchanged."""
    inputs = Input(shape=input_shape)
    if model_type == "GRU_Attention":
        x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = GRU(100, return_sequences=True, activation="relu")(x)
        x = Attention()([x, x])
        x = Dropout(0.3)(x)
        x = GRU(100, activation="relu")(x)
        x = RepeatVector(H)(x)
        x = GRU(100, return_sequences=True, activation="relu")(x)
        x = TimeDistributed(Dense(50, activation="relu"))(x)
    elif model_type == "LSTM":
        x = LSTM(128, return_sequences=False)(inputs)
        x = RepeatVector(H)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = TimeDistributed(Dense(50, activation="relu"))(x)
    elif model_type == "CNN_GRU":
        x = Conv1D(32, 3, activation="relu", padding="same")(inputs)
        x = GRU(64, return_sequences=False)(x)
        x = RepeatVector(H)(x)
        x = GRU(64, return_sequences=True)(x)
        x = TimeDistributed(Dense(50, activation="relu"))(x)
    else:
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(H)(x)
        x = Lambda(lambda t: tf.expand_dims(t, -1))(x)
    outputs = TimeDistributed(Dense(1))(x) if model_type != "Dense" else x
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


def windows(features, temps, times):
    """Inputs: hours [t-24, t). Targets: temperature for hours [t, t+24)."""
    X, y, starts = [], [], []
    for t in range(H, len(features) - H + 1):
        X.append(features[t - H:t])
        y.append(temps[t:t + H])
        starts.append(times[t])
    return np.array(X), np.array(y), starts


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "RMSE": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "R2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        "MAE_by_hour": [float(v) for v in np.mean(np.abs(y_true - y_pred), axis=0)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--residual", action="store_true",
                        help="train on the change from persistence; forecast = persistence + prediction")
    args = parser.parse_args()
    seed_everything()

    df = fetch()
    feature_cols = list(df.columns)  # includes past temperature_2m
    n = len(df)
    gap = 2 * H
    a, b = int(n * 0.70), int(n * 0.85)
    blocks = {"train": df.iloc[:a], "val": df.iloc[a + gap:b], "test": df.iloc[b + gap:]}

    scaler = StandardScaler().fit(blocks["train"][feature_cols])
    data = {}
    for name, block in blocks.items():
        X, y, starts = windows(scaler.transform(block[feature_cols]), block["temperature_2m"].to_numpy(), block.index)
        data[name] = (X, y, starts)
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    X_te, y_te, te_starts = data["test"]
    temp_col = feature_cols.index("temperature_2m")

    # Persistence: the next 24 hours repeat the previous 24 (inverse-scaled past temperature).
    def persistence(X):
        return X[:, :, temp_col] * scaler.scale_[temp_col] + scaler.mean_[temp_col]

    persist = persistence(X_te)
    results = {"Persistence (baseline)": metrics(y_te, persist)}
    predictions = {"Persistence (baseline)": persist}

    # In residual mode the networks learn only the correction to persistence.
    base_tr, base_va = (persistence(X_tr), persistence(X_va)) if args.residual else (0, 0)
    base_te = persist if args.residual else 0
    for model_type in MODELS:
        seed_everything()
        print(f"Training {model_type} ...", flush=True)
        model = build_model(model_type, X_tr.shape[1:])
        history = model.fit(X_tr, (y_tr - base_tr)[..., None], validation_data=(X_va, (y_va - base_va)[..., None]),
                            epochs=args.epochs, batch_size=32, verbose=0,
                            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        pred = model.predict(X_te, verbose=0).reshape(-1, H) + base_te
        results[model_type] = {**metrics(y_te, pred), "epochs_run": len(history.history["loss"])}
        predictions[model_type] = pred
        print(f"  {model_type}: MAE {results[model_type]['MAE']:.2f} R2 {results[model_type]['R2']:.3f}", flush=True)

    # The notebook's original evaluation, reproduced for GRU_Attention: same-time features
    # (no past temperature), windows i..i+24 -> i+1..i+25, shuffled split, 10 epochs.
    seed_everything()
    X_raw = StandardScaler().fit_transform(df.drop(columns=["temperature_2m"]))
    temps = df["temperature_2m"].to_numpy()
    Xo = np.array([X_raw[i:i + H] for i in range(len(X_raw) - H)])
    yo = np.array([temps[i + 1:i + 1 + H] for i in range(len(X_raw) - H)])
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=0.2, random_state=42)
    print("Training GRU_Attention with the original (leaky) evaluation ...", flush=True)
    model = build_model("GRU_Attention", Xo_tr.shape[1:])
    model.fit(Xo_tr, yo_tr[..., None], epochs=10, batch_size=32, verbose=0)
    original = metrics(yo_te, model.predict(Xo_te, verbose=0).reshape(-1, H))

    best = min(MODELS, key=lambda m: results[m]["MAE"])
    examples = []
    for idx in np.linspace(0, len(te_starts) - 1, 4, dtype=int):  # four forecasts spread across the test period
        start = te_starts[idx]
        past = X_te[idx, :, temp_col] * scaler.scale_[temp_col] + scaler.mean_[temp_col]
        examples.append({
            "forecast_start": start.isoformat(),
            "past_24h": [round(float(v), 2) for v in past],
            "actual_next_24h": [round(float(v), 2) for v in y_te[idx]],
            "predicted_next_24h": [round(float(v), 2) for v in predictions[best][idx]],
            "persistence_next_24h": [round(float(v), 2) for v in persist[idx]],
        })

    out = {
        "target": "change from persistence" if args.residual else "temperature",
        "data": {"location": "Ahmedabad (23.03 N, 72.59 E)", "from": str(df.index[0]), "to": str(df.index[-1]),
                 "hours": n, "features": feature_cols},
        "split": {name: {"from": str(block.index[0]), "to": str(block.index[-1]), "windows": len(data[name][0])}
                  for name, block in blocks.items()},
        "best_model": best,
        "results": results,
        "original_notebook_method_GRU_Attention": original,
        "examples": examples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("best:", best, "| wrote", args.out)


if __name__ == "__main__":
    main()
