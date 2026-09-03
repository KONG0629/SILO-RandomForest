"""S.I.L.O. Analytics: automated 10-minute Random Forest forecasts.

GitHub Actions reads sensor history from Supabase, trains time-aware models,
publishes one forecast per storage, and publishes active fan-control rules.
"""

import os
import sys
import warnings
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- Configuration -----------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. "
        "Add both values to GitHub repository Actions secrets."
    )

DEVICE_ID = os.getenv("SILO_DEVICE_ID", "ESP32-S3-SILO-001").strip()
SENSOR_TABLE = "sensor_readings"
PREDICTIONS_TABLE = "predictions"
MODEL_RULES_TABLE = "model_rules"
STORAGE_NUMBERS = (1, 2, 3)

FORECAST_MINUTES = 10
FORECAST_TOLERANCE_MINUTES = 4
MAX_SOURCE_ROWS = 50_000
PAGE_SIZE = 1_000
MAX_LATEST_AGE_MINUTES = 30
MIN_TRAINING_PAIRS = 30
RANDOM_STATE = 42

SENSOR_COLUMNS = ["temperature", "humidity", "mq135_raw"]
FEATURE_COLUMNS = [
    "storage_no",
    "temperature",
    "humidity",
    "mq135_raw",
    "temperature_delta",
    "humidity_delta",
    "mq135_delta",
]
RISK_ORDER = {"safe": 0, "warning": 1, "critical": 2}

DEFAULT_RULES = {
    "temperature_on": 30.0,
    "temperature_off": 28.0,
    "humidity_on": 70.0,
    "humidity_off": 65.0,
    "air_quality_on": 1500.0,
    "air_quality_off": 1300.0,
}

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


# ----------------------------- Supabase REST -----------------------------

def supabase_request(method, table, *, params=None, payload=None, prefer=None):
    headers = dict(HEADERS)
    if prefer:
        headers["Prefer"] = prefer

    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.request(
                method,
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers=headers,
                params=params,
                json=payload,
                timeout=(10, 60),
            )
            if response.ok:
                if response.status_code == 204 or not response.text.strip():
                    return []
                return response.json()

            body = response.text[:1000]
            last_error = RuntimeError(
                f"Supabase {method} {table} failed "
                f"({response.status_code}): {body}"
            )
            # Retrying invalid requests or authentication failures will not help.
            if response.status_code < 500 and response.status_code != 429:
                raise last_error
        except requests.RequestException as error:
            last_error = error

        if attempt < 3:
            import time
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Supabase request failed after 3 attempts: {last_error}")


def fetch_sensor_data():
    """Fetch sensor history for the configured device from Supabase."""
    selected = (
        "id,device_id,storage_no,temperature,humidity,"
        "mq135_raw,risk_label,created_at"
    )

    project_host = urlparse(SUPABASE_URL).netloc or SUPABASE_URL
    print(f"Supabase project: {project_host}")
    print(f"Sensor table: {SENSOR_TABLE}")
    print(f"Device ID query: {DEVICE_ID!r}")

    rows = []
    for offset in range(0, MAX_SOURCE_ROWS, PAGE_SIZE):
        page = supabase_request(
            "GET",
            SENSOR_TABLE,
            params={
                "device_id": f"eq.{DEVICE_ID}",
                "select": selected,
                "order": "created_at.desc,id.desc",
                "limit": str(PAGE_SIZE),
                "offset": str(offset),
            },
        )

        if not isinstance(page, list):
            raise TypeError(
                f"Expected a list from Supabase, received {type(page).__name__}."
            )

        print(f"Sensor page offset={offset}: {len(page)} row(s)")
        rows.extend(page)

        if len(page) < PAGE_SIZE:
            break

    if rows:
        frame = pd.DataFrame(rows)
        print(f"Downloaded {len(frame):,} sensor row(s) for {DEVICE_ID}.")
        return frame

    # Diagnostic query: this does not train on another device. It only reveals
    # whether GitHub is connected to the expected Supabase project/table.
    sample = supabase_request(
        "GET",
        SENSOR_TABLE,
        params={
            "select": "device_id,created_at",
            "order": "created_at.desc",
            "limit": "20",
        },
    )

    available_ids = sorted({
        str(row.get("device_id", "")).strip()
        for row in sample
        if isinstance(row, dict) and row.get("device_id")
    })

    print(f"No rows matched device ID {DEVICE_ID!r}.")
    print(f"Device IDs visible in this Supabase project: {available_ids or 'none'}")

    if DEVICE_ID in available_ids:
        print(
            "The correct ID exists in Supabase, but the filtered request returned "
            "nothing. Check the table permissions and service-role secret."
        )
    elif available_ids:
        print(
            "GitHub is reaching Supabase, but this project contains different "
            "device IDs. Check SILO_DEVICE_ID and the SUPABASE_URL secret."
        )
    else:
        print(
            "No sensor rows are visible. The GitHub secrets may point to another "
            "Supabase project, or the sensor_readings table is empty there."
        )

    return pd.DataFrame()


# -------------------------- Cleaning and pairing --------------------------

def clean_sensor_data(frame):
    required = {
        "id", "storage_no", "temperature", "humidity",
        "mq135_raw", "risk_label", "created_at",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {SENSOR_TABLE}: {missing}")

    clean = frame.copy()
    clean["id"] = pd.to_numeric(clean["id"], errors="coerce")
    for column in ["storage_no", *SENSOR_COLUMNS]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["created_at"] = pd.to_datetime(
        clean["created_at"], utc=True, errors="coerce"
    )
    clean["risk_label"] = (
        clean["risk_label"].astype("string").str.strip().str.lower()
        .replace({"normal": "safe"})
    )

    before_drop = len(clean)
    clean = clean.dropna(
        subset=["id", "storage_no", *SENSOR_COLUMNS, "risk_label", "created_at"]
    )
    print(
        f"Rows with all required sensor values: {len(clean):,}/{before_drop:,}"
    )
    clean["storage_no"] = clean["storage_no"].astype(int)
    clean = clean[
        clean["storage_no"].isin(STORAGE_NUMBERS)
        & clean["temperature"].between(-10, 80)
        & clean["humidity"].between(0, 100)
        & clean["mq135_raw"].between(0, 4095)
        & clean["risk_label"].isin(RISK_ORDER)
    ]
    clean = (
        clean.sort_values(["storage_no", "created_at", "id"])
        .drop_duplicates("id", keep="last")
    )

    for column in SENSOR_COLUMNS:
        delta_name = f"{column.replace('_raw', '')}_delta"
        clean[delta_name] = clean.groupby("storage_no")[column].diff().fillna(0.0)

    clean = clean.reset_index(drop=True)
    print(f"Usable cleaned sensor rows: {len(clean):,}")
    return clean


def build_forecast_pairs(clean):
    parts = []
    horizon = pd.Timedelta(minutes=FORECAST_MINUTES)
    tolerance = pd.Timedelta(minutes=FORECAST_TOLERANCE_MINUTES)

    for _, current in clean.groupby("storage_no", sort=True):
        current = current.sort_values("created_at").copy()
        current["target_time"] = current["created_at"] + horizon

        future = current[
            ["created_at", *SENSOR_COLUMNS, "risk_label"]
        ].rename(columns={
            "created_at": "future_created_at",
            "temperature": "future_temperature",
            "humidity": "future_humidity",
            "mq135_raw": "future_mq135_raw",
            "risk_label": "future_risk_label",
        })

        paired = pd.merge_asof(
            current.sort_values("target_time"),
            future.sort_values("future_created_at"),
            left_on="target_time",
            right_on="future_created_at",
            direction="nearest",
            tolerance=tolerance,
        )
        paired = paired.dropna(subset=[
            "future_created_at",
            "future_temperature",
            "future_humidity",
            "future_mq135_raw",
            "future_risk_label",
        ])
        # This explicit check prevents an unexpectedly old row becoming a target.
        gap = paired["future_created_at"] - paired["created_at"]
        paired = paired[
            gap.between(
                horizon - tolerance,
                horizon + tolerance,
                inclusive="both",
            )
        ]
        parts.append(paired)

    if not parts:
        return pd.DataFrame()
    pairs = pd.concat(parts, ignore_index=True).sort_values("created_at")
    print(f"Built {len(pairs):,} valid T+{FORECAST_MINUTES} training pairs.")
    return pairs.reset_index(drop=True)


# ------------------------------- Models ----------------------------------

def make_preprocessor():
    return ColumnTransformer(
        [("storage", OneHotEncoder(handle_unknown="ignore"), ["storage_no"])],
        remainder="passthrough",
    )


def make_regressor():
    return Pipeline([
        ("prepare", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def make_classifier():
    return Pipeline([
        ("prepare", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def train_models(pairs):
    if len(pairs) < MIN_TRAINING_PAIRS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_PAIRS} valid 10-minute pairs; "
            f"found {len(pairs)}."
        )

    ordered = pairs.sort_values("created_at").reset_index(drop=True)
    split_at = min(len(ordered) - 1, max(1, int(len(ordered) * 0.80)))
    train, test = ordered.iloc[:split_at], ordered.iloc[split_at:]
    targets = ["future_temperature", "future_humidity", "future_mq135_raw"]

    evaluation_regressor = make_regressor()
    evaluation_regressor.fit(train[FEATURE_COLUMNS], train[targets])
    reg_prediction = evaluation_regressor.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test[targets], reg_prediction, multioutput="raw_values")

    metrics = {
        "temperature_mae": float(mae[0]),
        "humidity_mae": float(mae[1]),
        "mq135_mae": float(mae[2]),
        "accuracy": None,
        "macro_f1": None,
        "test_rows": int(len(test)),
    }

    classifier = None
    if ordered["future_risk_label"].nunique() >= 2:
        # Evaluate only when the chronological training portion has two classes.
        if train["future_risk_label"].nunique() >= 2:
            evaluation_classifier = make_classifier()
            evaluation_classifier.fit(
                train[FEATURE_COLUMNS], train["future_risk_label"]
            )
            class_prediction = evaluation_classifier.predict(test[FEATURE_COLUMNS])
            metrics["accuracy"] = float(
                accuracy_score(test["future_risk_label"], class_prediction)
            )
            metrics["macro_f1"] = float(
                f1_score(
                    test["future_risk_label"],
                    class_prediction,
                    average="macro",
                    zero_division=0,
                )
            )

        classifier = make_classifier()
        classifier.fit(
            ordered[FEATURE_COLUMNS], ordered["future_risk_label"]
        )
    else:
        print(
            "Only one risk class is available. Sensor forecasts will still run; "
            "risk will be calculated from the predicted values and thresholds."
        )

    regressor = make_regressor()
    regressor.fit(ordered[FEATURE_COLUMNS], ordered[targets])
    print("Evaluation:", metrics)
    return regressor, classifier, metrics


# -------------------------- Rules and predictions -------------------------

def bounded(value, low, high):
    return float(np.clip(float(value), low, high))


def generate_model_rules(pairs):
    rules_by_storage = {}
    for storage_no in STORAGE_NUMBERS:
        storage = pairs[pairs["storage_no"] == storage_no]
        risky = storage[
            storage["future_risk_label"].isin(["warning", "critical"])
        ]

        if len(risky) < 5:
            rules = dict(DEFAULT_RULES)
            source = "default"
        else:
            temperature_on = bounded(
                risky["temperature"].quantile(0.25), 25.0, 30.0
            )
        
            humidity_on = bounded(
                risky["humidity"].quantile(0.25), 60.0, 70.0
            )
        
            air_quality_on = bounded(
                risky["mq135_raw"].quantile(0.25), 100.0, 4000.0
            )
        
            rules = {
                "temperature_on": temperature_on,
                "temperature_off": temperature_on - 2.0,
        
                "humidity_on": humidity_on,
                "humidity_off": humidity_on - 5.0,
        
                "air_quality_on": air_quality_on,
                "air_quality_off": air_quality_on - max(
                    50.0,
                    air_quality_on * 0.10
                ),
            }
        
            source = "random_forest"

        rules["source"] = source
        rules_by_storage[storage_no] = rules
    return rules_by_storage


def publish_model_rules(rules_by_storage, metrics, model_version):
    for storage_no, rule_with_source in rules_by_storage.items():
        rules = {
            key: round(float(value), 2)
            for key, value in rule_with_source.items()
            if key != "source"
        }
        payload = {
            "device_id": DEVICE_ID,
            "storage_no": storage_no,
            "model_version": model_version,
            **rules,
            "accuracy": metrics["accuracy"],
            "prediction_type": "risk_classification",
            "is_active": True,
            "notes": (
                f"source={rule_with_source['source']}; "
                f"macro_f1={metrics['macro_f1']}"
            ),
        }
        # Insert first, so a failed insert never leaves the ESP32 without a rule.
        supabase_request(
            "POST",
            MODEL_RULES_TABLE,
            payload=payload,
            prefer="return=representation",
        )
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
        print(f"Storage {storage_no}: model rule published.")


def fetch_latest_sensor_reading(storage_no):
    rows = supabase_request(
        "GET",
        SENSOR_TABLE,
        params={
            "device_id": f"eq.{DEVICE_ID}",
            "storage_no": f"eq.{storage_no}",
            "select": (
                "id,device_id,storage_no,temperature,humidity,"
                "mq135_raw,risk_label,created_at"
            ),
            "order": "created_at.desc,id.desc",
            "limit": 2,
        },
    )
    if not rows:
        return None

    latest = rows[0]
    latest_time = pd.to_datetime(
        latest.get("created_at"), utc=True, errors="coerce"
    )
    if pd.isna(latest_time):
        return None
    age = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() / 60
    if age > MAX_LATEST_AGE_MINUTES:
        print(f"Storage {storage_no}: latest reading is {age:.1f} minutes old.")
        return None

    previous = rows[1] if len(rows) > 1 else latest
    try:
        for sensor in SENSOR_COLUMNS:
            delta_name = f"{sensor.replace('_raw', '')}_delta"
            latest[delta_name] = float(latest[sensor]) - float(previous[sensor])
    except (KeyError, TypeError, ValueError):
        return None
    return latest


def risk_from_values(temperature, humidity, mq135, rules):
    risks = []
    if temperature >= rules["temperature_on"]:
        risks.append("high_temperature")
    if humidity >= rules["humidity_on"]:
        risks.append("high_humidity")
    if mq135 >= rules["air_quality_on"]:
        risks.append("poor_air_quality")

    count = len(risks)
    status = "critical" if count >= 2 else "warning" if count == 1 else "safe"
    return status, risks or ["none"]


def forecast_latest(latest, regressor, classifier, rules):
    features = pd.DataFrame([{
        "storage_no": int(latest["storage_no"]),
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "mq135_raw": float(latest["mq135_raw"]),
        "temperature_delta": float(latest["temperature_delta"]),
        "humidity_delta": float(latest["humidity_delta"]),
        "mq135_delta": float(latest["mq135_delta"]),
    }])[FEATURE_COLUMNS]

    values = regressor.predict(features)[0]

    temperature = bounded(values[0], -10, 80)
    humidity = bounded(values[1], 0, 100)
    mq135 = bounded(values[2], 0, 4095)

    threshold_status, sources = risk_from_values(
        temperature,
        humidity,
        mq135,
        rules
    )

    status = threshold_status
    confidence = None

    if classifier is not None:
        model_status = str(classifier.predict(features)[0]).lower()
        probability = classifier.predict_proba(features)[0]
        confidence = float(np.max(probability))

        if RISK_ORDER.get(model_status, 0) > RISK_ORDER[status]:
            status = model_status
            sources = [*sources, "risk_classifier"]

    prediction_for = (
        pd.to_datetime(latest["created_at"], utc=True)
        + pd.Timedelta(minutes=FORECAST_MINUTES)
    )

    # Convert individual risk conditions to text
    temperature_risk = (
        "high_temperature"
        if temperature >= rules["temperature_on"]
        else "normal"
    )

    humidity_risk = (
        "high_humidity"
        if humidity >= rules["humidity_on"]
        else "normal"
    )

    air_quality_risk = (
        "poor_air_quality"
        if mq135 >= rules["air_quality_on"]
        else "normal"
    )

    return {
        "prediction_for": prediction_for.isoformat(),

        "predicted_temperature": round(temperature, 2),
        "predicted_humidity": round(humidity, 2),
        "predicted_air_quality": round(mq135, 2),

        "prediction_status": status,
        "prediction_score": (
            round(confidence, 6)
            if confidence is not None
            else None
        ),

        "temperature_risk": temperature_risk,
        "humidity_risk": humidity_risk,
        "air_quality_risk": air_quality_risk,
    }

def fetch_storage_column_id(storage_no):
    rows = supabase_request(
        "GET",
        "storage_columns",
        params={
            "device_id": f"eq.{DEVICE_ID}",
            "column_number": f"eq.{int(storage_no)}",
            "select": "id,column_number,column_name",
            "limit": 1,
        },
    )

    if not rows:
        raise RuntimeError(
            f"No storage_columns row found for device "
            f"{DEVICE_ID!r}, column_number={storage_no}."
        )

    storage_column_id = rows[0].get("id")

    if storage_column_id is None:
        raise RuntimeError(
            f"storage_columns row for column {storage_no} has no id."
        )

    return int(storage_column_id)

def save_prediction(latest, forecast, model_version):
    storage_no = int(latest["storage_no"])

    # Get the real primary-key ID from storage_columns.
    storage_column_id = fetch_storage_column_id(storage_no)

    payload = {
        "device_id": DEVICE_ID,
        "storage_column_id": storage_column_id,
        "sensor_reading_id": int(latest["id"]),
        "storage_no": storage_no,

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
            "sensor_reading_id": f"eq.{int(latest['id'])}",
            "storage_no": f"eq.{storage_no}",
            "select": "id",
            "limit": 1,
        },
    )

    if existing:
        supabase_request(
            "PATCH",
            PREDICTIONS_TABLE,
            params={"id": f"eq.{existing[0]['id']}"},
            payload=payload,
        )
    else:
        supabase_request(
            "POST",
            PREDICTIONS_TABLE,
            payload=payload,
            prefer="return=representation",
        )

    print(
        f"Storage {storage_no}: prediction saved "
        f"(storage_column_id={storage_column_id})."
    )


# ------------------------------ Main run ---------------------------------

def run_pipeline():
    started = datetime.now(timezone.utc)
    model_version = "RF10M_" + started.strftime("%Y%m%d_%H%M%S")
    print(f"S.I.L.O. Random Forest run: {model_version}")

    raw = fetch_sensor_data()
    if raw.empty:
        raise RuntimeError(
            "No matching sensor data was returned by Supabase. "
            "Read the diagnostic lines above to check the project, table, "
            "device ID, and GitHub secrets."
        )

    clean = clean_sensor_data(raw)
    if clean.empty:
        raise RuntimeError(
            "Rows were downloaded, but none remained after validation. "
            "Check for NULL/invalid temperature, humidity, mq135_raw, "
            "risk_label, storage_no, or created_at values."
        )

    pairs = build_forecast_pairs(clean)
    if pairs.empty:
        raise RuntimeError(
            "Sensor rows were found, but no valid 10-minute training pairs "
            "could be built. Make sure readings cover more than 10 minutes."
        )
    regressor, classifier, metrics = train_models(pairs)
    rules_by_storage = generate_model_rules(pairs)

    publish_model_rules(rules_by_storage, metrics, model_version)

    saved = 0
    for storage_no in STORAGE_NUMBERS:
        latest = fetch_latest_sensor_reading(storage_no)
        if latest is None:
            print(f"Storage {storage_no}: no fresh valid reading; skipped.")
            continue
        forecast = forecast_latest(
            latest,
            regressor,
            classifier,
            rules_by_storage[storage_no],
        )
        save_prediction(latest, forecast, model_version)
        saved += 1

    print(
        f"Completed: {len(pairs)} pairs, {saved} predictions, "
        f"elapsed={(datetime.now(timezone.utc) - started).total_seconds():.1f}s"
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
