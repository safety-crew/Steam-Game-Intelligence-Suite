# Video Game Sales Intelligence Suite

A comprehensive machine learning project that analyzes video game sales data through three distinct analytical approaches: sales tier classification, regional sales prediction, and market trend clustering.

## 📊 Overview

This project tackles three machine learning tasks to extract valuable insights from video game sales data:

### 🎯 Task 1: Sales Tier Classification (Multi-class Classification)

Predict whether a game will be a **Flop / Mid-tier / Hit / Blockbuster** based on its features.

- **Features**: genre, console, publisher, developer, critic_score, release_date
- **Target**: bucketed total_sales (<0.5M, 0.5–2M, 2–10M, >10M)
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM

### 🌍 Task 2: Regional Sales Prediction (Regression)

Predict sales in North America, Japan, and PAL regions to understand regional market behavior.

- **Features**: genre, console, critic_score, total_sales
- **Target**: na_sales, jp_sales, pal_sales (separately)
- **Models**: Ridge Regression, Random Forest Regressor, Gradient Boosting, Neural Network (MLP)

### 🔍 Task 3: Genre/Console Trend Clustering (Unsupervised)

Discover natural groupings among games to surface hidden market segments.

- **Features**: aggregated sales by genre × console × era
- **Models**: K-Means, DBSCAN, Hierarchical Clustering

## 📁 Project Structure

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

## 📋 Dataset

The dataset contains comprehensive video game sales information with the following key fields:

- **Game Information**: title, console, genre, publisher, developer
- **Critical Reception**: critic_score (Metacritic score out of 10)
- **Sales Data**:
  - total_sales: Global sales in millions of copies
  - na_sales: North American sales
  - jp_sales: Japanese sales
  - pal_sales: European & African sales
  - other_sales: Rest of world sales
- **Temporal**: release_date, last_update

**Data Source**: VGChartz.com (aggregated sales data)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, lightgbm

### Installation

1. Clone this repository:

```bash
git clone https://github.com/safety-crew/MA-Video-Game-Sales-Intelligence-Suite.git
cd video-game-sales-intelligence
```

2. Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

### Usage

1. Start with data cleaning:

```bash
jupyter notebook notebooks/00_cleaning.ipynb
```

2. Follow the numbered notebooks in sequence for complete analysis

## 📈 Analysis Workflow

1. **Data Cleaning** (`00_cleaning.ipynb`): Handle missing values, data types, and prepare features
2. **Exploratory Data Analysis** (`01_eda.ipynb`): Understand data distributions and relationships
3. **Classification** (`02_classification.ipynb`): Build and compare sales tier prediction models
4. **Regression** (`03_regression.ipynb`): Develop regional sales prediction models
5. **Clustering** (`04_clustering.ipynb`): Identify market segments and trends
6. **Model Comparison** (`05_model_comparison.ipynb`): Evaluate and compare all model performances

## 🎯 Key Insights Expected

- Which features most influence game sales success
- Regional market preferences and differences
- Optimal release strategies by genre and platform
- Hidden market segments and emerging trends
- Model performance comparison across different ML approaches

## 🤝 Contributing

This project follows a structured ML workflow. Contributions should focus on:

- Improving model performance
- Adding new features or analysis techniques
- Enhancing data preprocessing
- Creating visualizations and insights

## 📄 License

This project is for educational and research purposes. Please refer to VGChartz terms of service for data usage.

---

**Note**: This project is currently in development. The cleaning notebook is not complete, with additional analysis notebooks planned for implementation.
