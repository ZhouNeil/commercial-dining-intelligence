# Dual-Mode Commercial & Dining Intelligence App

This repository contains the codebase for our Machine Learning course final project. The application utilizes the Yelp Open Dataset to provide insights for both prospective merchants and tourists.

## 📂 Repository Structure

Our codebase strictly follows the Separation of Concerns (SoC) principle. The directory tree below outlines our modular architecture:

```text
commercial-dining-intelligence/
├── app/                        # Streamlit UI & Controller Logic (Role 5)
│   └── main.py                 # Dual-mode entry point and API Glue Code (Controller)
├── data/                       # Local storage for datasets and models (Git-ignored)
│   ├── processed_csv/          # Cleaned, city-specific feature matrices (e.g., output_philly.csv)
│   └── saved_models/           # Serialized city-specific predictive models (e.g., lr_philly.pkl)
├── pipelines/                  # Data engineering, feature extraction & aggregation
│   ├── data_cleaner.py         # Transforms raw Yelp JSONs into structured matrices (Role 1)
│   ├── feature_pca.py          # Applies PCA for business DNA extraction (Role 5)
│   └── feature_aggregator.py   # Online engine to calculate density & avg ratings (Role 1)
├── models/                     # Core ML algorithms and predictive engines
│   ├── knn_scratch.py          # Pure NumPy k-NN Engine (Radius & Top-K) (Role 2)
│   ├── tourist_recommender.py  # Cross-modal embeddings & NLP representation (Role 3)
│   ├── merchant_predictor.py   # Merchant survival/rating classification & regression (Role 4)
│   └── rl_feedback_loop.py     # Multi-Armed Bandit environment & weight updates (Role 6)
├── notebooks/                  # Jupyter notebooks for EDA, PCA visualization, and MVP testing
├── scripts/                    # Shell scripts for environment setup and pipeline execution
├── .gitignore                  # Prevents data/ and large model files from being tracked
├── README.md                   # Project architecture, API contracts, and setup instructions
└── requirements.txt            # Locked dependency registry for CI/CD (Role 6)
```

## 📥 Data Setup

Due to the massive size of the Yelp Open Dataset, raw and cleaned data files are **not** tracked in this GitHub repository. To run this project, you must manually download the required CSV files.

**1. Download the Data:**
* Fetch the cleaned dataset (e.g., `output_philly.csv`) from our team's shared Google: [`https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing`](https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing)

**2. Place it in the Repo:**
* Move the downloaded files directly into the `data/processed_csv/` folder. Do not force-add data files to Git.

## 🚀 Setup Environment

We use `uv` for lightning-fast dependency management. To run this project locally, please follow these steps from the root directory of the project:

**1. Initial Setup (Run Once):**
This script will install `uv` (if necessary), create a virtual environment (`.venv`), and install all dependencies from `requirements.txt`.

```bash
source scripts/setup_env.sh
```

**2. Activate Environment (Run every time you code):**
Whenever you open a new terminal session to work on this project, activate the environment by running:

```bash
source scripts/activate_env.sh
```

If you do not want to use `uv`, you can also install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Starting the App

Make sure you have the cleaned CSVs under `data/cleaned/`:
- `business_dining.csv`
- `review_dining.csv`

On the first run, the app builds the retrieval index under `models/artifacts/` (may take a while).

Run Streamlit:

```bash
streamlit run app/main.py
```
