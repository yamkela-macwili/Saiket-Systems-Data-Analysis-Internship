from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # project root, wherever it is on your machine
RAW = ROOT / "data" / "raw" / "Telco_Customer_Churn_Dataset_.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
for d in (PROCESSED, FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)
