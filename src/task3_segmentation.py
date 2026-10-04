"""Task 3: Customer Segmentation - tenure, monthly charges, contract type."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

from config import PROCESSED, FIGURES, TABLES
df = pd.read_csv(PROCESSED / "telco_churn_cleaned.csv")
df["ChurnFlag"] = (df.Churn == "Yes").astype(int)
sns.set_theme(style="whitegrid")

# ---- Rule-based segments ----
df["TenureSegment"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72],
                             labels=["New (0-12m)", "Developing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"])
df["ChargeSegment"] = pd.qcut(df.MonthlyCharges, 3, labels=["Low", "Medium", "High"])
print("Monthly-charge tercile cut points ($):", df.MonthlyCharges.quantile([1/3, 2/3]).round(2).tolist())

def table(col):
    t = df.groupby(col, observed=True).agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"),
                                           churn_rate_pct=("ChurnFlag", lambda s: round(s.mean()*100, 1)))
    print(f"\nChurn by {col}:\n{t}"); return t

for c in ["TenureSegment", "ChargeSegment", "Contract"]: table(c)

# ---- Combined segment: tenure x contract, and tenure x charge ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=df, x="TenureSegment", y="ChurnFlag", errorbar=None, color="#4C78A8", ax=axes[0])
axes[0].set_title("Churn rate by tenure segment")
sns.barplot(data=df, x="ChargeSegment", y="ChurnFlag", errorbar=None, color="#F58518", ax=axes[1])
axes[1].set_title("Churn rate by monthly-charge segment")
sns.barplot(data=df, x="Contract", y="ChurnFlag", errorbar=None, color="#54A24B", ax=axes[2])
axes[2].set_title("Churn rate by contract type")
for a in axes:
    a.set_ylabel("Churn rate"); a.set_xlabel(""); a.tick_params(axis="x", rotation=15)
    for p in a.patches: a.annotate(f"{p.get_height():.1%}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIGURES / "seg_01_single_dimension.png", dpi=150); plt.close()

pivot = df.pivot_table(index="TenureSegment", columns="Contract", values="ChurnFlag", aggfunc="mean", observed=True) * 100
pivot2 = df.pivot_table(index="ChargeSegment", columns="Contract", values="ChurnFlag", aggfunc="mean", observed=True) * 100
print("\nChurn % - tenure x contract:\n", pivot.round(1)); print("\nChurn % - charge x contract:\n", pivot2.round(1))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=axes[0]); axes[0].set_title("Churn % : tenure x contract")
sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="Reds", ax=axes[1]); axes[1].set_title("Churn % : monthly charge x contract")
plt.tight_layout(); plt.savefig(FIGURES / "seg_02_heatmaps.png", dpi=150); plt.close()

# ---- RFM-style view (adapted: Recency not available in this dataset) ----
# Tenure ~ loyalty/recency proxy, TotalCharges ~ monetary value, count of services ~ engagement ("frequency")
svc = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[svc] == "Yes").sum(axis=1) + (df.InternetService != "No").astype(int)
df["T_score"] = pd.qcut(df.tenure.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
df["M_score"] = pd.qcut(df.TotalCharges.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
df["S_score"] = pd.qcut(df.NumServices.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
df["RFM_total"] = df[["T_score", "M_score", "S_score"]].sum(axis=1)
df["Value_Tier"] = pd.cut(df.RFM_total, [2, 4, 6, 9], labels=["Low value", "Mid value", "High value"])
t = df.groupby("Value_Tier", observed=True).agg(customers=("ChurnFlag", "size"),
        avg_monthly=("MonthlyCharges", "mean"), avg_tenure=("tenure", "mean"),
        churn_rate_pct=("ChurnFlag", lambda s: s.mean()*100)).round(1)
print("\nRFM-style value tiers:\n", t)

# Highest / lowest risk combined segments
df["Segment"] = df.TenureSegment.astype(str) + " | " + df.ChargeSegment.astype(str) + " charge | " + df.Contract
g = df.groupby("Segment").agg(customers=("ChurnFlag", "size"), churn_rate_pct=("ChurnFlag", lambda s: s.mean()*100)).round(1)
g = g[g.customers >= 50].sort_values("churn_rate_pct", ascending=False)
print("\nTop 5 riskiest combined segments (n>=50):\n", g.head(5)); print("\nSafest 5:\n", g.tail(5))
g.to_csv(TABLES / "segment_churn_table.csv")
