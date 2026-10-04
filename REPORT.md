# Customer Churn Analysis and Prediction
**Saiket Systems - Data Analysis Internship** | Prepared by: Yamkela Macwili | October 2026
**Dataset:** Telco Customer Churn, 7,043 customers, 21 variables | **Tasks completed:** 1, 2, 3, 4 (of 6)

## 1. Executive summary
About **1 in 4 customers (26.5%)** has churned. Churn is not spread evenly: it is concentrated among **new, month-to-month customers**, especially those on fiber-optic internet, paying by electronic check, without tech support or online security. A logistic-regression model ranks customers by churn risk well (**ROC-AUC 0.84**): the 20% of customers it scores as highest risk contain **51% of all churners** and churn at 2.5 times the base rate. Contract type is the strongest single factor (month-to-month 42.7% vs two-year 2.8%).

## 2. Data preparation (Task 1)
* 11 blank `TotalCharges` values, hidden because the column was read as text. All belong to customers with tenure 0 (not yet billed), so they were set to 0.
* "No internet service" / "No phone service" merged into "No" in seven service columns (redundant with `InternetService` / `PhoneService`).
* Binary variables mapped to 0/1; `InternetService`, `Contract` and `PaymentMethod` one-hot encoded (first level dropped). Result: 23 numeric features plus the `Churn` target, with row and churn counts validated.

## 3. Exploratory analysis (Task 2)
| Question | Finding |
|---|---|
| Overall churn | 26.5% (1,869 of 7,043) |
| Gender | No effect (26.9% vs 26.2%; p = 0.49) |
| Seniors / partner / dependents | Seniors 41.7% vs 23.6%; no partner 33.0% vs 19.7%; no dependents 31.3% vs 15.5% |
| Tenure | Churners' median tenure 10 months vs 38 for retained; churn 47.4% in year 1 vs 17.1% afterwards |
| Contract | Month-to-month 42.7%, one-year 11.3%, two-year 2.8% |
| Payment method | Electronic check 45.3% vs 15.2-19.1% for other methods |
| Internet service | Fiber optic 41.9%, DSL 19.0%, none 7.4% |
| Support add-ons | No tech support 31.2% vs 15.2%; no online security 31.3% vs 14.6% |

Chi-square tests with effect sizes (Cramer's V) rank contract (0.41), internet service (0.32) and payment method (0.30) as the strongest relationships.

## 4. Segmentation (Task 3)
Customers were segmented by tenure band, monthly-charge tercile and contract type, plus an RFM-style value tier (using tenure, total charges and number of services as proxies, since no recency data exists).
* Riskiest segment: **new + high charge + month-to-month, 74.6% churn** (370 customers).
* The four riskiest month-to-month segments hold **32% of customers but 63% of all churn**.
* Two-year contracts stay at or below about 5.5% churn in every tenure and price segment.
* RFM-style tiers: low value 38.8%, mid 25.1%, high 17.5% churn.

## 5. Prediction model (Task 4)
Four models (logistic regression, balanced logistic regression, random forest, gradient boosting) were compared on a stratified 80/20 split with 5-fold cross-validation. All reach **ROC-AUC about 0.84**, so the interpretable **logistic regression** was chosen.

| Metric (test set, threshold 0.5) | Value |
|---|---|
| Accuracy (baseline "nobody churns": 73.5%) | 0.807 |
| Precision / Recall / F1 (churn class) | 0.66 / 0.56 / 0.61 |
| ROC-AUC | 0.842 |

* **Drivers:** short tenure, month-to-month contract, fiber optic, higher total charges, streaming add-ons and electronic check raise risk; longer contracts and tenure lower it.
* **Threshold:** a business choice. At 0.30 the model catches 75% of churners (precision 0.52); at 0.50, 56% (precision 0.66).
* **Ranking power (out-of-fold):** actual churn rises from 1.3% in the lowest-risk decile to 76% in the highest.
* **Calibration:** close to actual by contract type (e.g., two-year 2.9% predicted vs 2.8% actual).

## 6. Recommendations
These follow from the findings above; they would need testing (for example A/B trials) before being treated as proven.
1. **Focus on the first year.** Nearly half of year-one customers churn. A structured onboarding and 30/60/90-day check-in programme targets the highest-risk window.
2. **Move month-to-month customers to longer contracts**, using incentives. Churn on one- and two-year contracts is far lower, though customers who choose long contracts may differ from those who do not.
3. **Investigate fiber-optic service.** Fiber customers churn at 41.9% versus 19.0% for DSL. Check price competitiveness, reliability and support experience for this product.
4. **Promote tech support and online security** as bundled add-ons for at-risk customers; both are associated with churn about half as high.
5. **Encourage automatic payment.** Electronic-check customers churn at 45.3%, about three times the rate of automatic payers.
6. **Use the model to prioritise outreach**, with the threshold set from the real cost of an offer and the value of a retained customer (the cost figures in the notebook are placeholders).

## 7. Limitations
* Single snapshot: the data does not show *when* customers leave or why.
* Findings show association, not cause (for example, electronic check may mark less engaged customers rather than cause churn).
* No cost or revenue data is supplied, so the cost model uses illustrative numbers only.
* Predictions for combinations that barely appear in the data (for example new customers on two-year contracts) are unreliable.
* Model performance would need monitoring on new data over time.

## 8. Files
`notebooks/01-04_*.ipynb` (one per task, executed), `data/processed/` (cleaned and encoded data), `outputs/figures/` (charts), `outputs/tables/` (result tables and per-customer risk scores).
