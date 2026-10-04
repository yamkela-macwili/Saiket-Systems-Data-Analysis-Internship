"""Task 2: Exploratory Data Analysis - Telco Customer Churn."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency

from config import PROCESSED, FIGURES, TABLES
df = pd.read_csv(PROCESSED / "telco_churn_cleaned.csv")
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
sns.set_theme(style="whitegrid")
PAL = {"No": "#4C78A8", "Yes": "#E45756"}

# ---- 1. Overall churn rate ----
rate = df.ChurnFlag.mean() * 100
print(f"Overall churn rate: {rate:.2f}%  ({df.ChurnFlag.sum()} of {len(df)})")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
counts = df.Churn.value_counts()
ax[0].pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[PAL[i] for i in counts.index], startangle=90)
ax[0].set_title("Overall churn rate")
sns.countplot(data=df, x="Churn", palette=PAL, hue="Churn", legend=False, ax=ax[1])
for p in ax[1].patches: ax[1].annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[1].set_title("Customers by churn status")
plt.tight_layout(); plt.savefig(FIGURES / "eda_01_overall_churn.png", dpi=150); plt.close()

# ---- 2. Demographics ----
demo = ["gender", "SeniorCitizen", "Partner", "Dependents"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, c in enumerate(demo):
    sns.countplot(data=df, x=c, color="#72B7B2", ax=axes[0, i]); axes[0, i].set_title(f"Customers by {c}")
    r = df.groupby(c).ChurnFlag.mean() * 100
    sns.barplot(x=r.index, y=r.values, color="#E45756", ax=axes[1, i]); axes[1, i].set_title(f"Churn rate (%) by {c}")
    for p in axes[1, i].patches: axes[1, i].annotate(f"{p.get_height():.1f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIGURES / "eda_02_demographics.png", dpi=150); plt.close()
print("\nChurn rate by demographic (%):")
for c in demo: print(df.groupby(c).ChurnFlag.mean().mul(100).round(1).to_dict(), "<-", c)

# ---- 3. Tenure distribution ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=36, palette=PAL, multiple="stack", ax=ax[0])
ax[0].set_title("Tenure distribution (months)")
sns.boxplot(data=df, x="Churn", y="tenure", palette=PAL, hue="Churn", legend=False, ax=ax[1])
ax[1].set_title("Tenure by churn status")
plt.tight_layout(); plt.savefig(FIGURES / "eda_03_tenure.png", dpi=150); plt.close()
print("\nMedian tenure - churned:", df[df.Churn=='Yes'].tenure.median(), "| retained:", df[df.Churn=='No'].tenure.median())
print("Churn rate in first 12 months: %.1f%% | after 12 months: %.1f%%" % (
    df[df.tenure<=12].ChurnFlag.mean()*100, df[df.tenure>12].ChurnFlag.mean()*100))

# ---- 4. Contract type & payment method ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, c in zip(ax, ["Contract", "PaymentMethod"]):
    r = df.groupby(c).ChurnFlag.mean().mul(100).sort_values()
    sns.barplot(x=r.values, y=r.index, color="#E45756", ax=a)
    a.set_title(f"Churn rate (%) by {c}"); a.set_ylabel("")
    for p in a.patches: a.annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y()+p.get_height()/2), va="center", ha="left")
    a.set_xlim(0, r.max()*1.2)
plt.tight_layout(); plt.savefig(FIGURES / "eda_04_contract_payment.png", dpi=150); plt.close()
print("\nChurn by Contract (%):\n", df.groupby("Contract").ChurnFlag.mean().mul(100).round(1))
print("\nChurn by PaymentMethod (%):\n", df.groupby("PaymentMethod").ChurnFlag.mean().mul(100).round(1))

# Statistical test: chi-square of independence vs churn
print("\nChi-square tests vs Churn (p-values):")
for c in ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]:
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[c], df.Churn))
    print(f"  {c:18s} p = {p:.2e}  {'significant' if p < 0.05 else 'not significant'}")
