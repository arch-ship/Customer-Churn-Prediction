# 📡 Telco Customer Churn Prediction & Customer Segmentation
### CBSOT Summer Internship 2026 | End-to-End Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-blueviolet)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit)](https://YOUR-APP-URL.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Live Demo →** [customer-churn-prediction-arch-ship.streamlit.app](https://customer-churn-prediction-arch-ship.streamlit.app/)  
> *(Replace with your Streamlit Community Cloud URL after deployment)*

| Notebook | Open in Colab |
|---|---|
| 📓 01 — EDA & Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch-ship/customer-churn-prediction/blob/main/notebook/01_EDA_and_Preprocessing.ipynb) |
| 📓 02 — Modeling, SHAP & Segmentation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch-ship/customer-churn-prediction/blob/main/notebook/02_Modeling_SHAP_Segmentation.ipynb) |

---

## 📌 Project Overview

Customer churn — when customers stop using a service — is one of the biggest challenges for subscription-based telecom businesses. **Retaining an existing customer is 5–25× cheaper than acquiring a new one.**

This project builds a **complete end-to-end ML platform** that:
1. **Predicts** whether a telecom customer is likely to churn (Binary Classification)
2. **Explains** *why* using SHAP (SHapley Additive exPlanations)
3. **Segments** customers into actionable business groups using K-Means Clustering
4. **Serves** predictions via an interactive Streamlit dashboard

---

## 🗂️ Repository Structure

```
customer-churn-prediction/
│
├── 📓 notebook/
│   ├── 01_EDA_and_Preprocessing.ipynb     # Data exploration & cleaning
│   └── 02_Modeling_SHAP_Segmentation.ipynb # Full ML pipeline
│
├── 🌐 app/
│   └── app.py                              # Streamlit dashboard
│
├── 📊 outputs/                             # Auto-generated charts & CSVs
│   ├── 01_churn_distribution.png
│   ├── 07_feature_importance_rf.png
│   ├── 11_shap_summary_beeswarm.png
│   ├── 16_cluster_scatter_plots.png
│   ├── model_comparison.csv
│   └── cluster_statistics.csv
│
├── 📁 data/
│   └── Telco_customer_churn.xlsx           # IBM Telco dataset (7,043 records)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | IBM Telco Customer Churn Dataset |
| **Total Records** | 7,043 customers |
| **Features** | 33 columns (demographics, services, billing) |
| **Target** | `Churn Value` (1 = Churned, 0 = Retained) |
| **Class Distribution** | 5,174 Retained (73.46%) / 1,869 Churned (26.54%) |
| **Key Challenge** | Class imbalance (3:1 ratio) |

---

## 🔧 ML Pipeline

### Step 1 — Exploratory Data Analysis (EDA)
- Churn distribution analysis (pie chart + countplot)
- Tenure, Monthly Charges, Total Charges distributions
- Contract type vs churn rate (crosstab + countplot)
- Internet Service, Tech Support, Payment Method vs churn
- Correlation heatmap of numerical features
- **Quartile analysis** of Monthly Charges separately for churned vs retained

### Step 2 — Data Preprocessing
- `pd.to_numeric()` on Total Charges (fixes object dtype for 11 zero-tenure customers)
- Drop irrelevant / target-leakage columns (CustomerID, Churn Score, CLTV, Churn Reason, geographic columns)
- **One-Hot Encoding** via `pd.get_dummies(drop_first=True)` — 20 categorical features encoded
- Low-importance feature removal post feature-importance analysis

### Step 3 — Train-Test Split
- 80/20 split | `random_state=42` | **stratified** to preserve class distribution

### Step 4 — Random Forest (3 Approaches)

| Approach | Config | Purpose |
|---|---|---|
| **Baseline** | `n_estimators=100, random_state=42` | Benchmark |
| **Class Balanced** | `class_weight='balanced'` | Improve recall for minority class |
| **GridSearchCV Tuned** | 5×5 grid = **25 combinations**, `scoring='recall'` | Optimize for churn recall |

GridSearchCV parameter grid:
```python
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth'   : [5, 10, 15, 20, 25]
}
# cv=3, scoring='recall' → 75 fits total
```

### Step 5 — XGBoost (Novel Addition)
- `scale_pos_weight = neg_count / pos_count` for class imbalance
- `n_estimators=300, max_depth=6, learning_rate=0.05`
- Side-by-side comparison with Random Forest

### Step 6 — Model Evaluation

All models evaluated on: Accuracy, Precision, **Recall (priority)**, F1-Score, ROC-AUC, 5-Fold Cross Validation

---

## 📈 Model Performance Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RF — Baseline | 0.7934 | 0.6421 | 0.5132 | 0.5702 | 0.8312 |
| RF — Class Balanced | 0.7821 | 0.5963 | 0.7184 | 0.6517 | 0.8487 |
| RF — GridSearchCV | 0.7798 | 0.5891 | 0.7401 | 0.6562 | 0.8531 |
| XGBoost | 0.7912 | 0.6284 | 0.7089 | 0.6664 | 0.8619 |

> **Why Recall > Accuracy?** In churn prediction, a **False Negative** (predicting "won't churn" when customer actually churns) costs the business a lost customer. We optimize for Recall to catch as many churners as possible.

**5-Fold Cross Validation (RF GridSearchCV):**
- CV Accuracy  : 0.779 ± 0.008
- CV Recall    : 0.733 ± 0.015
- CV F1        : 0.652 ± 0.011
- CV ROC-AUC   : 0.851 ± 0.009

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions at both **global** and **individual** level. Unlike default RF feature importance, SHAP accounts for feature interactions.

```python
explainer  = shap.TreeExplainer(rf_tuned)
shap_vals  = explainer.shap_values(X_test_sample)
```

**SHAP confirms top churn drivers:**
1. `Tenure Months` — Short tenure = high churn impact
2. `Monthly Charges` — High charges push toward churn
3. `Contract_Month-to-month` — Strongest categorical churn signal
4. `Internet Service_Fiber optic` — Higher churn risk
5. `Tech Support_No` — Absence of support increases churn

**Outputs generated:**
- `11_shap_summary_beeswarm.png` — Global feature impact with direction
- `12_shap_feature_importance_bar.png` — Mean |SHAP| bar chart
- `13_shap_force_plot_high_risk.html` — Individual prediction explanation

---

## 🗂️ Customer Segmentation

**Method:** K-Means Clustering on `[Tenure Months, Monthly Charges, Total Charges, Churn Probability]`

**K Selection:** Elbow Method + Silhouette Score (dual validation)

| K | Silhouette Score |
|---|---|
| 2 | 0.312 |
| **3** | **0.347 ← selected** |
| 4 | 0.298 |
| 5 | 0.271 |

### Customer Segments (K = 3)

| Segment | Count | Avg Tenure | Avg Monthly | Avg Churn Prob | Profile |
|---|---|---|---|---|---|
| 🟢 **Budget Loyal Customers** | ~484 | 33.3 months | $33.40 | 12.6% | Long tenure, low charges, loyal |
| 🔴 **High Risk Customers** | ~391 | 11.0 months | $72.21 | 68.0% | New customers, high charges, at risk |
| 🔵 **Loyal Premium Customers** | ~534 | 58.7 months | $90.99 | 22.7% | Long tenure, high value, stable |

**Churn Risk Categorization:**

| Risk Level | Threshold | Count | % |
|---|---|---|---|
| 🔴 High Risk | Prob ≥ 0.60 | ~1,162 | 16.5% |
| 🟡 Medium Risk | 0.30 ≤ Prob < 0.60 | ~1,847 | 26.2% |
| 🟢 Low Risk | Prob < 0.30 | ~4,034 | 57.3% |

---

## 💡 Key Business Insights

| Factor | Finding |
|---|---|
| **Contract Type** | Month-to-month: **42.7%** churn \| One-year: **11.3%** \| Two-year: **2.8%** |
| **Avg Tenure — Churned** | **17.98 months** vs 37.57 months (retained) — 2× difference |
| **Tech Support** | Customers without Tech Support churn **significantly more** |
| **Monthly Charges** | Churned customers pay on avg **$15+ more/month** than retained |
| **Internet Service** | Fiber optic users churn more than DSL — likely cost-driven |
| **Payment Method** | Electronic check users show highest churn rate |

### Business Recommendations per Segment

**🔴 High Risk Customers (Urgent):**
- Trigger retention campaigns within 7 days of identification
- Offer 3-month free Tech Support bundle
- Provide contract upgrade incentive (Month-to-Month → 1-year discount)
- Assign dedicated customer success manager

**🟢 Budget Loyal Customers:**
- Do NOT raise prices — price sensitivity is their key trait
- Reward loyalty milestones (6 months, 1 year, 2 years)
- Upsell affordable add-ons (Online Backup, Online Security)

**🔵 Loyal Premium Customers:**
- Priority customer support (dedicated helpline)
- Early access to new features and services
- Annual relationship review / VIP program enrollment

---

## 🛠️ Tech Stack

| Tool | Version | Use |
|---|---|---|
| Python | 3.10 | Core language |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical operations |
| Matplotlib | 3.9.0 | Visualizations |
| Seaborn | 0.13.2 | Statistical plots |
| Scikit-Learn | 1.5.0 | RF, KMeans, GridSearchCV, metrics |
| XGBoost | 2.0.3 | Gradient boosting model |
| SHAP | 0.45.1 | Model explainability |
| Streamlit | 1.35.0 | Interactive dashboard |
| OpenPyXL | 3.1.2 | Read .xlsx files |

---

## 🚀 How to Run

### Option 1: Notebooks (Local)
```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook
# Run notebook/01_EDA_and_Preprocessing.ipynb first
# Then run notebook/02_Modeling_SHAP_Segmentation.ipynb
```

### Option 2: Streamlit Dashboard (Local)
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Option 3: Live Demo
👉 [YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

---

## 📁 Output Files

After running the notebooks, the `outputs/` folder contains:

| File | Description |
|---|---|
| `01_churn_distribution.png` | Pie chart + countplot of churn |
| `02_tenure_analysis.png` | Tenure distribution + boxplot vs churn |
| `03_monthly_charges_analysis.png` | Charges distribution + boxplot |
| `04_contract_vs_churn.png` | Contract type churn rates |
| `05_categorical_vs_churn.png` | Internet, Payment, Tech Support |
| `06_correlation_heatmap.png` | Feature correlation matrix |
| `07_feature_importance_rf.png` | Top 20 RF feature importances |
| `08_model_comparison.png` | All model metrics bar charts |
| `09_roc_auc_curve.png` | RF vs XGBoost ROC curves |
| `10_confusion_matrix.png` | Confusion matrices |
| `11_shap_summary_beeswarm.png` | SHAP beeswarm plot |
| `12_shap_feature_importance_bar.png` | SHAP global bar chart |
| `13_shap_force_plot_high_risk.html` | Interactive SHAP force plot |
| `14_shap_xgboost_bar.png` | XGBoost SHAP importance |
| `15_elbow_and_silhouette.png` | K selection plots |
| `16_cluster_scatter_plots.png` | 3 cluster scatter plots |
| `17_segment_profile_heatmap.png` | Segment characteristic heatmap |
| `model_comparison.csv` | All model metrics table |
| `cluster_statistics.csv` | Exact cluster statistics |

---

## 🎓 Acknowledgements

- **Mentor:** Aryesh Rai Sir — for guidance throughout this project
- **CBSOT Team:** Kartik Mathur Sir, Varun Kohli Sir, Monu Kumar Sir
- **Organization:** Coding Blocks School of Technology
- **Dataset:** IBM Telco Customer Churn (via Kaggle)

---

## 🔮 Future Enhancements

- [ ] Deploy to Streamlit Community Cloud *(in progress)*
- [ ] Add SHAP waterfall plots per customer in the dashboard
- [ ] Try LightGBM and compare with XGBoost
- [ ] Add real-time API endpoint (FastAPI) for batch predictions
- [ ] Implement time-series churn forecasting

---

*Built with ❤️ during CBSOT Summer Internship 2026*
