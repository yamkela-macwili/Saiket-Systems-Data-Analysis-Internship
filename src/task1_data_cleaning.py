"""Task 1: Data Cleaning and Preprocessing - Telco Customer Churn."""
import pandas as pd

from config import RAW, PROCESSED, FIGURES, TABLES

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)

# ---- 1. Inspect quality ----
print("\nNull counts (before):\n", df.isna().sum()[df.isna().sum() > 0])
print("Duplicate rows:", df.duplicated().sum(), "| duplicate IDs:", df.customerID.duplicated().sum())

# ---- 2. Missing values ----
# TotalCharges is stored as text because 11 rows contain a blank string " ".
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
missing = df[df["TotalCharges"].isna()]
print(f"\nBlank TotalCharges rows: {len(missing)}; their tenure values: {missing['tenure'].unique()}")
# All have tenure == 0 (brand-new customers not yet billed), so the true total is 0.
df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
assert df.isna().sum().sum() == 0

# ---- 3. Consistency fixes ----
# "No internet service" / "No phone service" are redundant with InternetService / PhoneService == "No".
# Collapse to "No" so one-hot encoding doesn't create duplicate information.
service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
for c in service_cols:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})  # make Yes/No like other flags
df.to_csv(PROCESSED / "telco_churn_cleaned.csv", index=False)

# ---- 4. Encoding ----
enc = df.drop(columns="customerID").copy()
enc["Churn"] = enc["Churn"].map({"Yes": 1, "No": 0})
binary_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"] + service_cols
enc["gender"] = enc["gender"].map({"Female": 0, "Male": 1})
for c in binary_cols:
    if c != "gender":
        enc[c] = enc[c].map({"No": 0, "Yes": 1})
# Multi-category columns -> one-hot (drop_first avoids the dummy-variable trap)
enc = pd.get_dummies(enc, columns=["InternetService", "Contract", "PaymentMethod"],
                     drop_first=True, dtype=int)
enc.insert(0, "customerID", df["customerID"])
enc.to_csv(PROCESSED / "telco_churn_encoded.csv", index=False)

print("\nEncoded shape:", enc.shape)
print(list(enc.columns))
print("Dtypes all numeric:", enc.drop(columns='customerID').dtypes.map(lambda d: d.kind in 'iuf').all())
