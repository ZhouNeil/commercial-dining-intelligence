# Yelp Dual-Mode Commercial & Dining Intelligence App

This repository contains the codebase for our Machine Learning course final project. The application utilizes the Yelp Open Dataset to provide insights for both prospective merchants and tourists.

## 📂 Repository Structure

Our codebase strictly follows the Separation of Concerns (SoC) principle. The directory tree below outlines our modular architecture:

```text
commercial-dining-intelligence/
├── app/                        # Streamlit UI (main dashboard and visual components)
│   ├── main.py                 # Dual-mode web app entry point (Role 5)
│   └── components.py           # Reusable UI widgets like maps and charts (Role 5)
├── data/                       # Local storage for cleaned feature CSVs (Git-ignored)
├── pipelines/                  # Data engineering and feature extraction
│   ├── data_cleaner.py         # Transforms raw data into numerical matrices (Role 1)
│   └── feature_pca.py          # Applies PCA for business DNA extraction (Role 6)
├── models/                     # Core ML algorithms and predictive engines
│   ├── kmeans_scratch.py       # Pure NumPy K-Means clustering algorithm (Role 2)
│   ├── tourist_recommender.py  # Tourist Mode embedding & semantic search (Role 3)
│   ├── merchant_predictor.py   # Merchant Mode survival/rating prediction (Role 4)
│   └── rl_feedback_loop.py     # Reinforcement Learning & A/B testing weights (Role 6)
├── notebooks/                  # Jupyter notebooks for EDA and MVP testing
├── scripts/                    # Shell scripts for environment setup
├── .gitignore                  # Prevents data/ and .venv/ from being tracked
├── README.md                   # Project architecture and setup instructions
└── requirements.txt            # Locked dependency registry for CI/CD
```

## 📥 Data Setup

Due to the massive size of the Yelp Open Dataset, raw and cleaned data files are **not** tracked in this GitHub repository. To run this project, you must manually download the required CSV files.

**1. Download the Data:**
* Fetch the dataset from our team's shared Google Drive: [`https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing`](https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing)

**2. Place it in the Repo:**
* Move the downloaded files directly into the `data/` folder. Do not force-add data files to Git.

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
