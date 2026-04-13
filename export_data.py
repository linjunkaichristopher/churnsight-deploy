"""
Run this locally to export batch predictions, feature defaults, and model
into the deploy folder so the cloud API can work without MySQL or MLflow.

Usage:
    cd ~/Documents/bt4301/BT4301-Group4-Project-overall-v1-20260410
    source .venv/bin/activate
    python ~/Claude/deploy/export_data.py
"""
import shutil, os, json
import pandas as pd
from sqlalchemy import create_engine

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.expanduser(
    "~/Documents/bt4301/BT4301-Group4-Project-overall-v1-20260410"
)

# 1. Connect to MySQL
print("Connecting to MySQL...")
engine = create_engine(
    "mysql+pymysql://bt4301project:password@localhost:3306/bt4301_kkbox_db",
    pool_pre_ping=True,
)

# 2. Export batch_predictions → CSV
df = pd.read_sql("SELECT * FROM batch_predictions ORDER BY scoring_timestamp DESC LIMIT 5000", engine)
csv_path = os.path.join(DEPLOY_DIR, "batch_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Exported {len(df)} batch predictions → {csv_path}")

# 3. Export feature defaults (median for numeric, mode for categorical)
#    This is what the original serve.py uses to impute missing features.
print("Computing feature defaults from curated table...")
ref = pd.read_sql("SELECT * FROM curated_kkbox_final_features LIMIT 1000", engine)
defaults = {}
for col in ref.columns:
    if col in ("msno", "is_churn", "curated_timestamp", "_row_hash",
               "batch_id", "ingestion_timestamp", "source_table"):
        continue
    if pd.api.types.is_numeric_dtype(ref[col]):
        val = ref[col].median()
        defaults[col] = float(val) if pd.notna(val) else 0.0
    else:
        mode = ref[col].mode()
        defaults[col] = str(mode.iloc[0]) if not mode.empty else "unknown"

defaults_path = os.path.join(DEPLOY_DIR, "feature_defaults.json")
with open(defaults_path, "w") as f:
    json.dump(defaults, f, indent=2)
print(f"Exported {len(defaults)} feature defaults → {defaults_path}")

# 4. Copy the model bundle
src_model = os.path.join(PROJECT_ROOT, "artifacts", "best_model.joblib")
dst_model = os.path.join(DEPLOY_DIR, "best_model.joblib")
shutil.copy2(src_model, dst_model)
print(f"Copied model → {dst_model}")

# 5. Copy dashboard HTML into static/
static_dir = os.path.join(DEPLOY_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
src_html = os.path.join(os.path.dirname(DEPLOY_DIR), "churnsight_dashboard.html")
dst_html = os.path.join(static_dir, "index.html")
shutil.copy2(src_html, dst_html)
print(f"Copied dashboard → {dst_html}")

print("\n✅ Deploy folder is ready! Files:")
for f in sorted(os.listdir(DEPLOY_DIR)):
    full = os.path.join(DEPLOY_DIR, f)
    if os.path.isdir(full):
        for sf in os.listdir(full):
            size = os.path.getsize(os.path.join(full, sf))
            print(f"  {f}/{sf}  ({size:,} bytes)")
    else:
        size = os.path.getsize(full)
        print(f"  {f}  ({size:,} bytes)")
