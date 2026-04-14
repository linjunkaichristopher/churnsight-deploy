"""
Standalone FastAPI serving endpoint for KKBOX churn prediction.
Cloud-deployable version — uses a bundled model (.joblib) and
a CSV of batch predictions instead of MLflow + MySQL.
"""
import os
import json
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
BATCH_CSV = os.path.join(os.path.dirname(__file__), "batch_predictions.csv")
FEATURE_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "feature_defaults.json")
REGISTERED_MODEL_NAME = "kkbox-churn-model"

app = FastAPI(
    title="KKBOX Churn Prediction API",
    description="Cloud-deployed churn prediction for KKBOX subscribers.",
    version="1.0.0",
)

# CORS — allow the GitHub Pages frontend and any origin for the demo
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "https://bbbbronya.github.io,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for demo; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the dashboard HTML at root ────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/")
def serve_dashboard():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "KKBOX Churn API is running. POST to /predict-features to score customers."}

# ── Load model once ────────────────────────────────────────
_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
        _bundle.setdefault("model_name", REGISTERED_MODEL_NAME)
        _bundle.setdefault("threshold", 0.55)
        _bundle.setdefault("model_version", "1")
        _bundle.setdefault("run_id", "cloud-deploy")
        _bundle.setdefault("model_alias", "production")
        print(f"Loaded model: {_bundle['model_name']} threshold={_bundle['threshold']}")
    return _bundle


# ── Load batch predictions CSV once ────────────────────────
_batch_df = None

def get_batch_df():
    global _batch_df
    if _batch_df is None:
        if os.path.exists(BATCH_CSV):
            _batch_df = pd.read_csv(BATCH_CSV)
            print(f"Loaded {len(_batch_df)} batch predictions from CSV")
        else:
            _batch_df = pd.DataFrame()
            print("No batch_predictions.csv found — batch endpoints will return empty.")
    return _batch_df


# ── Load feature defaults (medians/modes from training data) ──
_feature_defaults = None

def get_feature_defaults():
    global _feature_defaults
    if _feature_defaults is None:
        if os.path.exists(FEATURE_DEFAULTS_PATH):
            with open(FEATURE_DEFAULTS_PATH) as f:
                _feature_defaults = json.load(f)
            print(f"Loaded {len(_feature_defaults)} feature defaults")
        else:
            _feature_defaults = {}
            print("No feature_defaults.json found — will use 0 for missing features.")
    return _feature_defaults


# ── Helpers ────────────────────────────────────────────────
def _risk_label(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


def _prediction_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    def _opt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return str(value)

    for _, row in df.iterrows():
        rows.append({
            "msno": _opt(row.get("msno")),
            "predicted_label": int(row["predicted_label"]),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "model_version": _opt(row.get("model_version")),
            "production_model_version": _opt(row.get("production_model_version")),
            "production_run_id": _opt(row.get("production_run_id")),
            "model_alias": _opt(row.get("model_alias")),
            "threshold_used": float(row["threshold_used"]) if row.get("threshold_used") is not None and not pd.isna(row.get("threshold_used")) else None,
            "scoring_timestamp": _opt(row.get("scoring_timestamp")),
        })
    return rows


def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    drop_cols = ["is_churn", "msno", "curated_timestamp", "_row_hash",
                 "batch_id", "ingestion_timestamp", "source_table"]
    feature_frame = df.drop(columns=drop_cols, errors="ignore")

    bundle = get_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]
    probs = model.predict_proba(feature_frame)[:, 1]
    labels = (probs >= threshold).astype(int)

    scored = df.copy()
    scored["predicted_label"] = labels
    scored["churn_probability"] = probs
    scored["model_version"] = bundle.get("model_version")
    scored["production_model_version"] = bundle.get("model_version")
    scored["production_run_id"] = bundle.get("run_id")
    scored["model_alias"] = bundle.get("model_alias", "production")
    scored["threshold_used"] = threshold
    return scored


# ── Pydantic models ────────────────────────────────────────
class PredictionRow(BaseModel):
    msno: str | None = None
    predicted_label: int
    churn_probability: float
    model_version: str | None = None
    production_model_version: str | None = None
    production_run_id: str | None = None
    model_alias: str | None = None
    threshold_used: float | None = None
    scoring_timestamp: str | None = None

class PredictionListResponse(BaseModel):
    count: int
    predictions: list[PredictionRow]

class DashboardSummaryResponse(BaseModel):
    total_predictions: int
    high_risk: int
    medium_risk: int
    low_risk: int
    avg_churn_probability: float
    positive_rate: float
    model_name: str
    model_alias: str
    generated_at: str
    risk_distribution: dict[str, int]
    probability_buckets: list[dict]

class PredictRawFeaturesRequest(BaseModel):
    features: list[dict]
    write_to_mysql: bool = False


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": REGISTERED_MODEL_NAME}


@app.get("/dashboard-summary", response_model=DashboardSummaryResponse)
def dashboard_summary():
    df = get_batch_df()
    if df.empty:
        return DashboardSummaryResponse(
            total_predictions=0, high_risk=0, medium_risk=0, low_risk=0,
            avg_churn_probability=0.0, positive_rate=0.0,
            model_name=REGISTERED_MODEL_NAME, model_alias="production",
            generated_at=pd.Timestamp.utcnow().isoformat(),
            risk_distribution={"high": 0, "medium": 0, "low": 0},
            probability_buckets=[],
        )

    probs = df["churn_probability"].astype(float)
    risk_series = probs.apply(_risk_label)
    rd = risk_series.value_counts().to_dict()

    edges = [(i/10, (i+1)/10) for i in range(10)]
    buckets = []
    for s, e in edges:
        mask = (probs >= s) & (probs <= e) if e == 1.0 else (probs >= s) & (probs < e)
        buckets.append({"bucket": f"{s:.1f}-{e:.1f}", "count": int(mask.sum())})

    bundle = get_bundle()
    return DashboardSummaryResponse(
        total_predictions=len(df),
        high_risk=rd.get("high", 0),
        medium_risk=rd.get("medium", 0),
        low_risk=rd.get("low", 0),
        avg_churn_probability=round(float(probs.mean()), 4),
        positive_rate=round(float((df["predicted_label"] == 1).mean()), 4),
        model_name=bundle.get("model_name", REGISTERED_MODEL_NAME),
        model_alias=bundle.get("model_alias", "production"),
        generated_at=pd.Timestamp.utcnow().isoformat(),
        risk_distribution={"high": rd.get("high", 0), "medium": rd.get("medium", 0), "low": rd.get("low", 0)},
        probability_buckets=buckets,
    )


@app.get("/predictions/latest", response_model=PredictionListResponse)
def latest_predictions(limit: int = 100):
    df = get_batch_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No predictions available.")

    if "scoring_timestamp" in df.columns:
        df = df.sort_values("scoring_timestamp", ascending=False)
    df = df.head(limit)

    if "predicted_label" not in df.columns or "churn_probability" not in df.columns:
        df = _score_dataframe(df)

    rows = [PredictionRow(**r) for r in _prediction_rows(df)]
    return PredictionListResponse(count=len(rows), predictions=rows)


def _engineer_features(raw: dict) -> dict:
    """
    Build all 37 model-ready features from a small set of raw user inputs.

    The frontend only needs to send the basic fields the user fills in.
    This function derives all time-windowed and aggregate features that
    the model expects, matching the patterns seen in the training data.
    """
    pc   = float(raw.get("payment_count", 0))
    tp   = float(raw.get("total_payment", 0))
    plp  = float(raw.get("plan_list_price", 149))
    aap  = float(raw.get("actual_amount_paid", 149))
    arr  = float(raw.get("auto_renew_rate", 0))
    cr   = float(raw.get("cancel_rate", 0))
    due  = float(raw.get("days_until_expire", 0))
    dsr  = float(raw.get("days_since_registration", 0))
    city = float(raw.get("city", 1))
    bd   = float(raw.get("bd", 0))
    rv   = float(raw.get("registered_via", 4))
    pmid = float(raw.get("payment_method_id", 1))

    return {
        # Direct inputs
        "payment_count": pc,
        "total_payment": tp,
        "plan_list_price": plp,
        "actual_amount_paid": aap,
        "auto_renew_rate": arr,
        "cancel_rate": cr,
        "days_until_expire": due,
        "days_since_registration": dsr,
        "city": city,
        "bd": bd,
        "registered_via": rv,
        "payment_method_id": pmid,
        # Derived aggregates
        "avg_payment": tp / pc if pc > 0 else 0,
        "auto_renew_count": round(pc * arr),
        "cancel_count": round(pc * cr),
        "transaction_recency_days": max(dsr - pc * 30, 0),
        # 30d/60d/90d windows — set to 0 to match training data pattern
        "payment_count_30d": 0, "total_payment_30d": 0,
        "payment_count_60d": 0, "total_payment_60d": 0,
        "payment_count_90d": 0, "total_payment_90d": 0,
        "auto_renew_count_30d": 0, "auto_renew_count_60d": 0, "auto_renew_count_90d": 0,
        "cancel_count_30d": 0, "cancel_count_60d": 0, "cancel_count_90d": 0,
        "auto_renew_rate_30d": 0, "auto_renew_rate_60d": 0, "auto_renew_rate_90d": 0,
        "cancel_rate_30d": 0, "cancel_rate_60d": 0, "cancel_rate_90d": 0,
        # Trends (no change between windows)
        "payment_trend_30_60": 0,
        "payment_trend_30_90": 0,
        "cancel_trend_30_90": 0,
        # Date & demographic features the model also expects
        "last_transaction_date": "2017-02-28",
        "membership_expire_date": "2017-04-09",
        "gender": "Unknown",
    }


@app.post("/predict-features")
def predict_features(request: PredictRawFeaturesRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="features list cannot be empty")

    raw_rows = request.features
    msno_list = [r.get("msno") for r in raw_rows]

    # Check if this is a full-feature CSV upload (has 30d/60d/90d columns)
    # vs a simple form submission (only basic fields)
    sample = raw_rows[0]
    is_full_feature = "payment_count_30d" in sample or "auto_renew_count_30d" in sample

    if is_full_feature:
        # CSV upload with all features — use as-is
        df = pd.DataFrame(raw_rows)
    else:
        # Simple form inputs — engineer all features server-side
        engineered = [_engineer_features(r) for r in raw_rows]
        df = pd.DataFrame(engineered)

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid feature rows provided")

    # Drop non-feature columns
    feature_frame = df.drop(
        columns=["msno", "is_churn", "curated_timestamp", "_row_hash", "batch_id",
                 "ingestion_timestamp", "source_table", "scoring_timestamp",
                 "predicted_label", "churn_probability", "threshold_used",
                 "model_version", "production_run_id", "name", "customer_name"],
        errors="ignore",
    )

    if feature_frame.empty or len(feature_frame.columns) == 0:
        raise HTTPException(status_code=400, detail="No valid features found.")

    bundle = get_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]

    expected = list(getattr(model, "feature_names_in_", []))
    defaults = get_feature_defaults()

    if expected:
        feature_frame = feature_frame.drop(
            columns=[c for c in feature_frame.columns if c not in expected],
            errors="ignore",
        )
        for col in expected:
            if col not in feature_frame.columns:
                feature_frame[col] = defaults.get(col, 0)
        feature_frame = feature_frame[expected]

    # Fill any remaining NaN with training-data defaults
    for col in feature_frame.columns:
        if feature_frame[col].isna().any():
            feature_frame[col] = feature_frame[col].fillna(defaults.get(col, 0))

    try:
        probs = model.predict_proba(feature_frame)[:, 1]
        labels = (probs >= threshold).astype(int)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    predictions = []
    for i, (prob, label) in enumerate(zip(probs, labels)):
        msno_val = str(msno_list[i]) if msno_list[i] is not None else f"new_customer_{i}"
        predictions.append(PredictionRow(
            msno=msno_val,
            predicted_label=int(label),
            churn_probability=round(float(prob), 4),
            model_version=str(bundle.get("model_version")),
            production_model_version=str(bundle.get("model_version")),
            production_run_id=str(bundle.get("run_id")),
            model_alias=str(bundle.get("model_alias", "production")),
            threshold_used=float(threshold),
            scoring_timestamp=pd.Timestamp.utcnow().isoformat(),
        ))

    return PredictionListResponse(count=len(predictions), predictions=predictions)


@app.get("/model-info")
def model_info():
    bundle = get_bundle()
    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": bundle.get("model_version"),
        "run_id": bundle.get("run_id"),
        "tags": {},
    }
