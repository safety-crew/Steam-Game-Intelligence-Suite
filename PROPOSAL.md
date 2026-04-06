# Video Game Sales Intelligence Suite

The idea is to tackle 3 distinct ML tasks, each using multiple models, then compare them.

---

## Task 1 - Sales Tier Classification (Multi-class Classification)

Predict whether a game is a **Flop / Mid / Hit / Blockbuster** based on its features.

- **Features:** genre, console, publisher, developer, critic_score, release_date.
- **Target:** bucketed `total_sales` (e.g., <0.5M, 0.5–2M, 2–10M, >10M).
- **Models to compare:** Logistic Regression, Random Forest, XGBoost, LightGBM.

---

## Task 2 - Regional Sales Prediction (Regression)

Predict `na_sales`, `jp_sales`, and `pal_sales` separately to understand regional market behavior.

- **Features:** genre, console, critic_score, total_sales (for one region, predict others).
- **Target:** each regional sales column.
- **Models to compare:** Ridge Regression, Random Forest Regressor, Gradient Boosting, Neural Network (MLP).

---

## Task 3 - Genre/Console Trend Clustering (Unsupervised)

Discover natural groupings among games to surface hidden market segments.

- **Features:** aggregated sales by genre × console × era
- **Models to compare:** K-Means, DBSCAN, Hierarchical Clustering

---

## Project Structure

```
├── README.md                     # Project documentation
├── PROPOSAL.md                   # Detailed project proposal
├── data/
│   └── raw/
│       ├── data.csv              # Main video game sales dataset
│       └── data_dictionary.csv   # Data field descriptions
└── notebooks/
    ├── 00_cleaning.ipynb         # Data cleaning and preprocessing
    ├── 01_eda.ipynb              # Exploratory data analysis (planned)
    ├── 02_classification.ipynb   # Sales tier classification (planned)
    ├── 03_regression.ipynb       # Regional sales prediction (planned)
    ├── 04_clustering.ipynb       # Market trend clustering (planned)
    └── 05_model_comparison.ipynb # Model comparison & results (planned)
```

---

## Why this works well as a multi-model project

- You get to use **classification, regression, and unsupervised** methods — covers a wide ML breadth
- The dataset has both numerical (`critic_score`, sales) and categorical (`genre`, `console`) features — good for feature engineering practice
- The ~64k rows is large enough to show meaningful differences between models
- Missing `critic_score` values (~30–40% likely) make imputation strategy a real design decision
