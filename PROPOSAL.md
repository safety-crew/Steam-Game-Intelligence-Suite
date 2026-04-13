# Steam Game Intelligence Suite

The idea is to tackle 3 distinct ML tasks, each using multiple models, then compare them.

---

## Task 1 - Review Score Classification (Multi-class Classification)

Predict a game's review sentiment category based on its features.

- **Target:** review score bucket derived from positive review ratio - **Overwhelmingly Negative / Mixed / Mostly Positive / Overwhelmingly Positive**
- **Features:** price, genre, tags, developer, DLC count, platform support, release year
- **Models to compare:** Logistic Regression, Random Forest, XGBoost, LightGBM

---

## Task 2 - Average Playtime Prediction (Regression)

Predict the average hours played per user to understand what drives long-term engagement.

- **Target:** average playtime (hours) - log-transformed to handle heavy right skew
- **Features:** genre, tags, price, review score, DLC count, estimated owners
- **Models to compare:** Ridge Regression, Random Forest Regressor, Gradient Boosting, Neural Network (MLP)

---

## Task 3 - Game Market Segmentation (Unsupervised)

Discover natural groupings among games to surface hidden market segments and genre niches.

- **Features:** price, playtime, review score, genre, tags, estimated owners
- **Models to compare:** K-Means, DBSCAN, Hierarchical Clustering

---

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

---

## Why this works well as a multi-model project

- You get to use **classification, regression, and unsupervised** methods - covers a wide ML breadth
- The dataset has numerical (`price`, `playtime`, `review counts`) and categorical (`genre`, `tags`) features - good for feature engineering practice
- Review score is derived from a ratio, making target engineering a real design decision
- Playtime is heavily right-skewed - log transformation and robust models become meaningful choices
- Tags are multi-label and high-cardinality - opens up interesting encoding strategies (multi-hot, embeddings, TF-IDF)
