# Steam Game Intelligence Suite

A comprehensive machine learning project that analyzes Steam game data through three distinct analytical approaches: review score classification, playtime prediction, and market segmentation clustering.

## Overview

This project tackles three machine learning tasks to extract valuable insights from Steam game data:

### Task 1: Review Score Classification (Multi-class Classification)

Predict a game's review sentiment category - **Overwhelmingly Negative / Mixed / Mostly Positive / Overwhelmingly Positive** - based on its features.

- **Features**: price, genre, tags, developer, DLC count, platform support, release year
- **Target**: review score bucket (derived from positive review ratio)
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM

### Task 2: Average Playtime Prediction (Regression)

Predict the average hours played per user for a game to understand what drives long-term engagement.

- **Features**: genre, tags, price, review score, DLC count, estimated owners
- **Target**: average playtime (hours, log-transformed)
- **Models**: Ridge Regression, Random Forest Regressor, Gradient Boosting, Neural Network (MLP)

### Task 3: Game Market Segmentation (Unsupervised)

Discover natural groupings among games to surface hidden market segments and genre niches.

- **Features**: price, playtime, review score, genre, tags, estimated owners
- **Models**: K-Means, DBSCAN, Hierarchical Clustering

## Project Structure

```
├── README.md                     # Project documentation
├── PROPOSAL.md                   # Detailed project proposal
├── data/
│   └── raw/
│       ├── MANIFEST.json                  # Dataset metadata
│       ├── applications.csv               # Core game records
│       ├── reviews.csv                    # Review scores and counts
│       ├── genres.csv                     # Genre lookup
│       ├── application_genres.csv         # Game-genre mapping
│       ├── categories.csv                 # Category lookup
│       ├── application_categories.csv     # Game-category mapping
│       ├── developers.csv                 # Developer lookup
│       ├── application_developers.csv     # Game-developer mapping
│       ├── publishers.csv                 # Publisher lookup
│       ├── application_publishers.csv     # Game-publisher mapping
│       ├── platforms.csv                  # Platform lookup
│       └── application_platforms.csv      # Game-platform mapping
└── notebooks/
    ├── 00_cleaning.ipynb         # Data cleaning and preprocessing
    ├── 01_eda.ipynb              # Exploratory data analysis (planned)
    ├── 02_classification.ipynb   # Review score classification (planned)
    ├── 03_regression.ipynb       # Playtime prediction (planned)
    ├── 04_clustering.ipynb       # Market segmentation (planned)
    └── 05_model_comparison.ipynb # Model comparison & results (planned)
```

## Dataset

The dataset is structured as a set of normalized relational tables:

| File                         | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| `applications.csv`           | Core game records - name, price, review scores, playtime, owners, release date |
| `reviews.csv`                | Per-game review data - positive/negative counts, review score category         |
| `genres.csv`                 | Genre lookup table                                                             |
| `application_genres.csv`     | Many-to-many mapping of games to genres                                        |
| `categories.csv`             | Category lookup table (e.g. Single-player, Co-op, VR)                          |
| `application_categories.csv` | Many-to-many mapping of games to categories                                    |
| `developers.csv`             | Developer lookup table                                                         |
| `application_developers.csv` | Many-to-many mapping of games to developers                                    |
| `publishers.csv`             | Publisher lookup table                                                         |
| `application_publishers.csv` | Many-to-many mapping of games to publishers                                    |
| `platforms.csv`              | Platform lookup table (Windows, Mac, Linux)                                    |
| `application_platforms.csv`  | Many-to-many mapping of games to platforms                                     |
| `MANIFEST.json`              | Dataset metadata                                                               |

**Data Source**: ![Steam Store / SteamSpy](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics)

## Getting Started

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

## Analysis Workflow

1. **Data Cleaning** (`00_cleaning.ipynb`): Handle missing values, data types, and prepare features
2. **Exploratory Data Analysis** (`01_eda.ipynb`): Understand data distributions and relationships
3. **Classification** (`02_classification.ipynb`): Build and compare review score prediction models
4. **Regression** (`03_regression.ipynb`): Develop average playtime prediction models
5. **Clustering** (`04_clustering.ipynb`): Identify market segments and genre niches
6. **Model Comparison** (`05_model_comparison.ipynb`): Evaluate and compare all model performances

## Key Insights Expected

- Which features most influence a game's review sentiment
- What drives long-term player engagement and high playtime
- How pricing, genre, and tags interact with player reception
- Hidden market segments and niche communities on Steam
- Model performance comparison across different ML approaches

## Contributing

This project follows a structured ML workflow. Contributions should focus on:

- Improving model performance
- Adding new features or analysis techniques
- Enhancing data preprocessing
- Creating visualizations and insights

## License

This project is for educational and research purposes. Please refer to Steam and SteamSpy terms of service for data usage.

---

**Note**: This project is currently in development. The cleaning notebook is in progress, with additional analysis notebooks planned for implementation.
