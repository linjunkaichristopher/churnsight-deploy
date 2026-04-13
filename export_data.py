"""
Run this locally to export batch predictions from MySQL into a CSV
that the cloud deployment can serve without a database.

Usage:
    cd ~/Documents/bt4301/BT4301-Group4-Project-overall-v1-20260410
    source .venv/bin/activate
    python ~/Claude/deploy/export_data.py
"""
import shutil, os
import pandas as pd
from sqlalchemy import create_engine

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Export batch_predictions from MySQL → CSV
print("Connecting to MySQL...")
engine = create_engine(
    "mysql+pymysql://bt4301project:password@localhost:3306/bt4301_kkbox_db",
    pool_pre_ping=True,
)
df = pd.read_sql("SELECT * FROM batch_predictions ORDER BY scoring_timestamp DESC LIMIT 5000", engine)
csv_path = os.path.join(DEPLOY_DIR, "batch_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Exported {len(df)} batch predictions → {csv_path}")

# 2. Copy the model bundle
PROJECT_ROOT = os.path.expanduser(
    "~/Documents/bt4301/BT4301-Group4-Project-overall-v1-20260410"
)
src_model = os.path.join(PROJECT_ROOT, "artifacts", "best_model.joblib")
dst_model = os.path.join(DEPLOY_DIR, "best_model.joblib")
shutil.copy2(src_model, dst_model)
print(f"Copied model → {dst_model}")

# 3. Copy dashboard HTML into static/
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
