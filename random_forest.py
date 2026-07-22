# ============================================================
# CELL 1: INSTALL AND IMPORT LIBRARIES
# ============================================================
# Run this cell once whenever a new Colab runtime starts.


import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)
print("✅ Libraries loaded.")

# ============================================================
# CONFIGURATION FOR GITHUB ACTIONS
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in GitHub Actions Secrets.")

DEVICE_ID = "ESP32_SILO_001"
SENSOR_TABLE = "sensordata"
PREDICTIONS_TABLE = "predictions"
MODEL_RULES_TABLE = "model_rules"
STORAGE_NUMBERS = [1, 2, 3]

FORECAST_MINUTES = 10
FORECAST_TOLERANCE_MINUTES = 4
MAX_SOURCE_ROWS = 50_000
PAGE_SIZE = 1_000
MAX_LATEST_AGE_MINUTES = 30
MIN_TRAINING_PAIRS = 30
MIN_CLASS_COUNT_FOR_TEST = 2
RANDOM_STATE = 42

SENSOR_COLUMNS = ["temperature", "humidity", "mq135_raw"]
FEATURE_COLUMNS = ["storage_no", "temperature", "humidity", "mq135_raw", "temperature_delta", "humidity_delta", "mq135_delta"]
VALID_LABELS = ["normal", "warning", "critical"]

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

print("✅ Configuration loaded for", DEVICE_ID)

# ============================================================
# CELL 3: SUPABASE REST HELPERS AND PAGINATED SENSOR FETCH
# ============================================================

def supabase_request(method, table, *, params=None, payload=None, prefer=None, timeout=30):
    headers = dict(HEADERS)
    if prefer:
        headers["Prefer"] = prefer
    response = requests.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        body = response.text[:1000]
        raise RuntimeError(f"Supabase {method} {table} failed ({response.status_code}): {body}")
    if response.status_code == 204 or not response.text.strip():
        return []
    return response.json()


def fetch_sensor_data():
    selected = "id,device_id,storage_no,temperature,humidity,mq135_raw,risk_label,created_at"
    rows = []

    for offset in range(0, MAX_SOURCE_ROWS, PAGE_SIZE):
        page = supabase_request(
            "GET",
            SENSOR_TABLE,
            params={
                "device_id": f"eq.{DEVICE_ID}",
                "select": selected,
                "order": "created_at.asc,id.asc",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break

    if len(rows) == MAX_SOURCE_ROWS:
        print(f"⚠️ Reached MAX_SOURCE_ROWS={MAX_SOURCE_ROWS}; older/newer coverage depends on query order.")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No sensor data found.")
        return df

    print(f"✅ Downloaded {len(df):,} sensor rows.")
    print(df["storage_no"].value_counts(dropna=False).sort_index())
    return df

# ============================================================
# CELL 4: CLEAN DATA AND BUILD TRUE T+10-MINUTE TRAINING PAIRS
# ============================================================

def clean_sensor_data(df):
    required = {"id", "storage_no", "temperature", "humidity", "mq135_raw", "risk_label", "created_at"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {SENSOR_TABLE}: {missing}")

    clean = df.copy()
    for column in ["storage_no", *SENSOR_COLUMNS]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["created_at"] = pd.to_datetime(clean["created_at"], utc=True, errors="coerce")
    clean["risk_label"] = clean["risk_label"].astype(str).str.strip().str.lower()
    clean = clean.dropna(subset=["id", "storage_no", *SENSOR_COLUMNS, "risk_label", "created_at"])
    clean["storage_no"] = clean["storage_no"].astype(int)
    clean = clean[
        clean["storage_no"].isin(STORAGE_NUMBERS)
        & clean["temperature"].between(-10, 80)
        & clean["humidity"].between(0, 100)
        & clean["mq135_raw"].between(0, 4095)
        & clean["risk_label"].isin(VALID_LABELS)
    ]
    clean = clean.sort_values(["storage_no", "created_at", "id"]).drop_duplicates("id", keep="last")

    # Recent change features help the forest learn whether conditions are rising or falling.
    for column in SENSOR_COLUMNS:
        clean[f"{column.replace('_raw', '')}_delta"] = clean.groupby("storage_no")[column].diff().fillna(0.0)

    return clean.reset_index(drop=True)


def build_forecast_pairs(clean):
    paired_parts = []
    horizon = pd.Timedelta(minutes=FORECAST_MINUTES)
    tolerance = pd.Timedelta(minutes=FORECAST_TOLERANCE_MINUTES)

    for storage_no, current in clean.groupby("storage_no", sort=True):
        current = current.sort_values("created_at").copy()
        current["target_time"] = current["created_at"] + horizon

        future = current[["created_at", *SENSOR_COLUMNS, "risk_label"]].rename(columns={
            "created_at": "future_created_at",
            "temperature": "future_temperature",
            "humidity": "future_humidity",
            "mq135_raw": "future_mq135_raw",
            "risk_label": "future_risk_label",
        }).sort_values("future_created_at")

        paired = pd.merge_asof(
            current.sort_values("target_time"),
            future,
            left_on="target_time",
            right_on="future_created_at",
            direction="nearest",
            tolerance=tolerance,
        )
        paired = paired.dropna(subset=["future_created_at", "future_temperature", "future_humidity", "future_mq135_raw", "future_risk_label"])
        paired["forecast_gap_minutes"] = (paired["future_created_at"] - paired["created_at"]).dt.total_seconds() / 60.0
        paired_parts.append(paired)

    if not paired_parts:
        return pd.DataFrame()
    pairs = pd.concat(paired_parts, ignore_index=True).sort_values("created_at").reset_index(drop=True)
    print(f"✅ Built {len(pairs):,} T+{FORECAST_MINUTES}-minute training pairs.")
    print(pairs.groupby("storage_no").size())
    print("Future risk labels:")
    print(pairs["future_risk_label"].value_counts())
    return pairs

# ============================================================
# CELL 5: TRAIN AND TIME-ORDER EVALUATE RANDOM FORESTS
# ============================================================

def make_preprocessor():
    return ColumnTransformer(
        [("storage", OneHotEncoder(handle_unknown="ignore"), ["storage_no"])],
        remainder="passthrough",
    )


def make_regressor():
    return Pipeline([
        ("prepare", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=350, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def make_classifier():
    return Pipeline([
        ("prepare", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def train_models(pairs):
    if len(pairs) < MIN_TRAINING_PAIRS:
        raise ValueError(f"Need at least {MIN_TRAINING_PAIRS} valid forecast pairs; found {len(pairs)}.")
    if pairs["future_risk_label"].nunique() < 2:
        raise ValueError("At least two future risk classes are required.")

    ordered = pairs.sort_values("created_at").reset_index(drop=True)
    split_at = max(1, int(len(ordered) * 0.80))
    train = ordered.iloc[:split_at]
    test = ordered.iloc[split_at:]
    if test.empty:
        raise ValueError("Not enough rows for a time-ordered test set.")

    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    target_columns = ["future_temperature", "future_humidity", "future_mq135_raw"]

    evaluation_regressor = make_regressor()
    evaluation_regressor.fit(X_train, train[target_columns])
    reg_pred = evaluation_regressor.predict(X_test)
    mae_values = mean_absolute_error(test[target_columns], reg_pred, multioutput="raw_values")

    evaluation_classifier = make_classifier()
    evaluation_classifier.fit(X_train, train["future_risk_label"])
    class_pred = evaluation_classifier.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(test["future_risk_label"], class_pred)),
        "macro_f1": float(f1_score(test["future_risk_label"], class_pred, average="macro", zero_division=0)),
        "temperature_mae": float(mae_values[0]),
        "humidity_mae": float(mae_values[1]),
        "mq135_mae": float(mae_values[2]),
        "test_rows": int(len(test)),
    }
    print("✅ Time-ordered evaluation:", metrics)
    print(classification_report(test["future_risk_label"], class_pred, zero_division=0))

    # Evaluation is finished; train production models on every approved pair.
    regressor = make_regressor()
    classifier = make_classifier()
    regressor.fit(ordered[FEATURE_COLUMNS], ordered[target_columns])
    classifier.fit(ordered[FEATURE_COLUMNS], ordered["future_risk_label"])
    return regressor, classifier, metrics

# ============================================================
# CELL 6: CREATE DATA-BASED, STORAGE-SPECIFIC SAFE THRESHOLDS
# ============================================================
# Thresholds use the lower quartile of CURRENT readings that were
# followed by warning/critical conditions about 10 minutes later.
# Safety bounds prevent the system from learning very unsafe values as normal.

DEFAULT_RULES = {
    "temperature_on": 30.0, "temperature_off": 28.0,
    "humidity_on": 70.0, "humidity_off": 65.0,
    "mq135_on": 1500.0, "mq135_off": 1300.0,
}

def _bounded(value, low, high):
    return float(np.clip(float(value), low, high))


def generate_model_rules(pairs):
    rules_by_storage = {}
    for storage_no in STORAGE_NUMBERS:
        storage = pairs[pairs["storage_no"] == storage_no]
        risky = storage[storage["future_risk_label"].isin(["warning", "critical"])]

        if len(risky) < 5:
            rules = dict(DEFAULT_RULES)
            source = f"defaults (only {len(risky)} future-risk rows)"
        else:
            temperature_on = _bounded(risky["temperature"].quantile(0.25), 25.0, 30.0)
            humidity_on = _bounded(risky["humidity"].quantile(0.25), 60.0, 70.0)
            mq135_on = _bounded(risky["mq135_raw"].quantile(0.25), 100.0, 4000.0)
            rules = {
                "temperature_on": temperature_on,
                "temperature_off": max(-10.0, temperature_on - 2.0),
                "humidity_on": humidity_on,
                "humidity_off": max(0.0, humidity_on - 5.0),
                "mq135_on": mq135_on,
                "mq135_off": max(0.0, mq135_on - max(50.0, mq135_on * 0.10)),
            }
            source = f"{len(risky)} future-risk rows"

        rules_by_storage[storage_no] = {k: round(float(v), 2) for k, v in rules.items()}
        print(f"Storage {storage_no}: {rules_by_storage[storage_no]} [{source}]")
    return rules_by_storage

# ============================================================
# CELL 7: SAFELY PUBLISH NEW MODEL RULES TO SUPABASE
# ============================================================
# Inserts the new active rule first. Only after a successful insert does it
# deactivate older versions, preventing a failed insert from leaving no rule.

def save_model_rules_to_supabase(rules_by_storage, metrics):
    model_version = "RF10M_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_saved = True

    for storage_no, rules in rules_by_storage.items():
        payload = {
            "device_id": DEVICE_ID,
            "storage_no": int(storage_no),
            "model_version": model_version,
            **rules,
            "accuracy": metrics.get("accuracy"),
            "prediction_type": "10_minute_sensor_and_risk_forecast",
            "is_active": True,
            "notes": f"Time-ordered RF; macro_f1={metrics.get('macro_f1', 0):.4f}",
        }
        try:
            supabase_request("POST", MODEL_RULES_TABLE, payload=payload, prefer="return=representation")
            supabase_request(
                "PATCH",
                MODEL_RULES_TABLE,
                params={
                    "device_id": f"eq.{DEVICE_ID}",
                    "storage_no": f"eq.{storage_no}",
                    "is_active": "eq.true",
                    "model_version": f"neq.{model_version}",
                },
                payload={"is_active": False},
            )
            print(f"✅ Storage {storage_no}: published {model_version}.")
        except Exception as error:
            all_saved = False
            print(f"❌ Storage {storage_no}: rule publication failed: {error}")

    return all_saved, model_version

# ============================================================
# CELL 8: GET AND VALIDATE THE LATEST READING FOR EACH STORAGE
# ============================================================

def fetch_latest_sensor_reading(storage_no):
    rows = supabase_request(
        "GET",
        SENSOR_TABLE,
        params={
            "device_id": f"eq.{DEVICE_ID}",
            "storage_no": f"eq.{storage_no}",
            "select": "id,device_id,storage_no,temperature,humidity,mq135_raw,risk_label,created_at",
            "order": "created_at.desc,id.desc",
            "limit": 2,
        },
    )
    if not rows:
        print(f"⚠️ Storage {storage_no}: no reading found.")
        return None

    latest = rows[0]
    latest_time = pd.to_datetime(latest["created_at"], utc=True, errors="coerce")
    age_minutes = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() / 60.0
    if pd.isna(latest_time) or age_minutes > MAX_LATEST_AGE_MINUTES:
        print(f"⚠️ Storage {storage_no}: newest reading is stale ({age_minutes:.1f} minutes old); skipped.")
        return None

    previous = rows[1] if len(rows) > 1 else latest
    try:
        latest["temperature_delta"] = float(latest["temperature"]) - float(previous["temperature"])
        latest["humidity_delta"] = float(latest["humidity"]) - float(previous["humidity"])
        latest["mq135_delta"] = float(latest["mq135_raw"]) - float(previous["mq135_raw"])
    except (TypeError, ValueError, KeyError) as error:
        print(f"⚠️ Storage {storage_no}: invalid latest values ({error}); skipped.")
        return None
    return latest

# ============================================================
# CELL 9: MAKE A T+10-MINUTE FORECAST AND IDENTIFY RISKS
# ============================================================

def forecast_latest(latest, regressor, classifier, rules):
    X = pd.DataFrame([{
        "storage_no": int(latest["storage_no"]),
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "mq135_raw": float(latest["mq135_raw"]),
        "temperature_delta": float(latest["temperature_delta"]),
        "humidity_delta": float(latest["humidity_delta"]),
        "mq135_delta": float(latest["mq135_delta"]),
    }])[FEATURE_COLUMNS]

    predicted_values = regressor.predict(X)[0]
    predicted_temperature = _bounded(predicted_values[0], -10, 80)
    predicted_humidity = _bounded(predicted_values[1], 0, 100)
    predicted_mq135 = _bounded(predicted_values[2], 0, 4095)
    predicted_status = str(classifier.predict(X)[0])
    probabilities = classifier.predict_proba(X)[0]
    confidence = float(np.max(probabilities))
    prediction_for = pd.to_datetime(latest["created_at"], utc=True) + pd.Timedelta(minutes=FORECAST_MINUTES)

    temperature_risk = predicted_temperature >= rules["temperature_on"]
    humidity_risk = predicted_humidity >= rules["humidity_on"]
    air_quality_risk = predicted_mq135 >= rules["mq135_on"]
    sources = []
    if temperature_risk: sources.append("high_temperature")
    if humidity_risk: sources.append("high_humidity")
    if air_quality_risk: sources.append("poor_air_quality")

    return {
        "prediction_for": prediction_for.isoformat(),
        "predicted_temperature": round(predicted_temperature, 2),
        "predicted_humidity": round(predicted_humidity, 2),
        "predicted_mq135_raw": round(predicted_mq135, 2),
        "prediction_status": predicted_status,
        "prediction_score": round(confidence, 6),
        "temperature_risk": bool(temperature_risk),
        "humidity_risk": bool(humidity_risk),
        "air_quality_risk": bool(air_quality_risk),
        "risk_sources": sources or ["none"],
    }

# ============================================================
# CELL 10: UPSERT/UPDATE A PREDICTION WITHOUT DUPLICATES
# ============================================================
# Required predictions columns include the old current-value columns plus:
# predicted_temperature, predicted_humidity, predicted_mq135_raw,
# prediction_for, and created_at (created_at may have a DB default).

def save_prediction_to_supabase(latest, forecast, model_version):
    payload = {
        "device_id": DEVICE_ID,
        "sensor_data_id": latest["id"],
        "storage_no": int(latest["storage_no"]),
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "mq135_raw": float(latest["mq135_raw"]),
        **forecast,
        "model_version": model_version,
    }

    existing = supabase_request(
        "GET",
        PREDICTIONS_TABLE,
        params={
            "sensor_data_id": f"eq.{latest['id']}",
            "storage_no": f"eq.{int(latest['storage_no'])}",
            "select": "id",
            "limit": 1,
        },
    )
    if existing:
        supabase_request("PATCH", PREDICTIONS_TABLE, params={"id": f"eq.{existing[0]['id']}"}, payload=payload)
        action = "updated"
    else:
        supabase_request("POST", PREDICTIONS_TABLE, payload=payload, prefer="return=representation")
        action = "inserted"
    print(f"✅ Storage {latest['storage_no']}: prediction {action}.")

# ============================================================
# CELL 11: COMPLETE ONE-RUN PIPELINE
# ============================================================

def run_silo_random_forest_pipeline(publish=True):
    print("=" * 60)
    print(f"S.I.L.O. ANALYTICS — {FORECAST_MINUTES}-MINUTE FORECAST")
    print("=" * 60)

    raw = fetch_sensor_data()
    if raw.empty:
        raise RuntimeError("Pipeline stopped: Supabase returned no sensor data.")

    clean = clean_sensor_data(raw)
    pairs = build_forecast_pairs(clean)
    regressor, classifier, metrics = train_models(pairs)
    rules_by_storage = generate_model_rules(pairs)

    model_version = "DRY_RUN_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if publish:
        rules_saved, model_version = save_model_rules_to_supabase(rules_by_storage, metrics)
        if not rules_saved:
            print("⚠️ One or more rules failed to publish; predictions will still be attempted.")

    saved_predictions = 0
    for storage_no in STORAGE_NUMBERS:
        latest = fetch_latest_sensor_reading(storage_no)
        if latest is None:
            continue
        forecast = forecast_latest(latest, regressor, classifier, rules_by_storage[storage_no])
        print(f"Storage {storage_no} forecast:", forecast)
        if publish:
            save_prediction_to_supabase(latest, forecast, model_version)
            saved_predictions += 1

    result = {
        "model_version": model_version,
        "training_pairs": len(pairs),
        "metrics": metrics,
        "rules": rules_by_storage,
        "predictions_saved": saved_predictions,
        "published": bool(publish),
    }
    print("✅ Pipeline completed.")
    return result

if __name__ == "__main__":
    run_silo_random_forest_pipeline(publish=True)
