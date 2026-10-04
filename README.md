# Customer Churn Analysis and Prediction

Telco churn project (Tasks 1-4): cleaning, EDA, segmentation, prediction model.

## Structure
```
telco-churn-project/
├── data/
│   ├── raw/            <- original CSV (input)
│   └── processed/      <- cleaned + encoded CSVs (created by task 1)
├── notebooks/
│   └── 01-04_*.ipynb          <- one executed notebook per task
├── src/
│   ├── config.py       <- all file paths
│   ├── task1_data_cleaning.py
│   ├── task2_eda.py
│   ├── task3_segmentation.py
│   ├── task4_churn_model.py
│   └── run_all.py
├── outputs/
│   ├── figures/        <- PNG charts
│   └── tables/         <- result CSVs
├── REPORT.md / VIDEO_SCRIPT.md
├── requirements.txt
└── README.md
```

## Setup (VS Code terminal, from the project folder)
Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then in VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> choose `.venv`.

## Run
```
python src/run_all.py            # everything
python src/task1_data_cleaning.py   # or one task at a time (run task 1 first)
```

## Notebooks (one per task)
Run them in order: Task 1 creates the files the others read.
| Notebook | Task |
|---|---|
| `notebooks/01_data_cleaning_preprocessing.ipynb` | Task 1 - Data cleaning and preprocessing |
| `notebooks/02_exploratory_data_analysis.ipynb` | Task 2 - EDA |
| `notebooks/03_customer_segmentation.ipynb` | Task 3 - Customer segmentation |
| `notebooks/04_churn_prediction_model.ipynb` | Task 4 - Churn prediction model |

Open each in VS Code, pick the `.venv` kernel (top-right) and use Run All. The notebooks are already executed, so outputs are visible before you run anything.
Also included: `REPORT.md` (written report) and `VIDEO_SCRIPT.md` (video outline and LinkedIn post draft).
