"""Run tasks 1-4 in order."""
import runpy
from pathlib import Path
here = Path(__file__).parent
for s in ["task1_data_cleaning", "task2_eda", "task3_segmentation", "task4_churn_model"]:
    print(f"\n{'='*20} {s} {'='*20}")
    runpy.run_path(str(here / f"{s}.py"), run_name="__main__")
