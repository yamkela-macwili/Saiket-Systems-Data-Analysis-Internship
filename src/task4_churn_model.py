"""Task 4: Churn Prediction Model - logistic regression (+ random forest comparison)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, classification_report)

from config import PROCESSED, FIGURES, TABLES
SEED = 42
enc = pd.read_csv(PROCESSED / "telco_churn_encoded.csv")
X, y = enc.drop(columns=["customerID", "Churn"]), enc["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train {X_tr.shape}, Test {X_te.shape}; churn share train {y_tr.mean():.3f} / test {y_te.mean():.3f}")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Logistic Regression (balanced)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
}
rows, fitted, probs = [], {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, m in models.items():
    m.fit(X_tr, y_tr); p = m.predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    fitted[name], probs[name] = m, p
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, p),
                 "CV ROC-AUC (5-fold, train)": cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()})
res = pd.DataFrame(rows).set_index("Model").round(3)
print("\n", res.to_string()); res.to_csv(TABLES / "model_comparison.csv")
print("\nBaseline (always predict 'No churn') accuracy: %.3f" % (1 - y_te.mean()))

# ---- Main model: logistic regression, standard threshold ----
main = "Logistic Regression"
pred = (probs[main] >= 0.5).astype(int)
print(f"\n=== {main} ===\n", classification_report(y_te, pred, target_names=["Retained", "Churned"]))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues", ax=ax[0],
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
ax[0].set_title(f"Confusion matrix - {main}"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
for name in models:
    fpr, tpr, _ = roc_curve(y_te, probs[name]); ax[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, probs[name]):.3f})")
ax[1].plot([0, 1], [0, 1], "k--", alpha=.4); ax[1].set_title("ROC curves"); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].legend(loc="lower right", fontsize=8)
coefs = pd.Series(fitted[main][-1].coef_[0], index=X.columns).sort_values()
top = pd.concat([coefs.head(8), coefs.tail(8)])
top.plot.barh(color=["#4C78A8" if v < 0 else "#E45756" for v in top], ax=ax[2])
ax[2].set_title("Top logistic-regression coefficients\n(standardised; red = raises churn risk)")
plt.tight_layout(); plt.savefig(FIGURES / "model_01_evaluation.png", dpi=150); plt.close()
print("Strongest churn drivers (+):\n", coefs.sort_values(ascending=False).head(6).round(2))
print("Strongest retention drivers (-):\n", coefs.sort_values().head(6).round(2))

# ---- Threshold trade-off for the business (catch more churners) ----
print("\nThreshold trade-off (logistic regression):")
for t in [0.5, 0.4, 0.3]:
    pr = (probs[main] >= t).astype(int)
    print(f"  threshold {t}: precision {precision_score(y_te, pr):.3f}  recall {recall_score(y_te, pr):.3f}  F1 {f1_score(y_te, pr):.3f}")

# ---- Save churn risk scores for every customer (useful for retention work) ----
final = fitted[main].fit(X, y)
enc[["customerID"]].assign(churn_probability=final.predict_proba(X)[:, 1].round(4)).to_csv(TABLES / "churn_risk_scores.csv", index=False)
