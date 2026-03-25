# Yelp Dual-Mode Commercial & Dining Intelligence App

This repository contains the codebase for our Machine Learning course final project. The application utilizes the Yelp Open Dataset to provide insights for both prospective merchants and tourists.

## 📂 Repository Structure

* `app/`: Streamlit frontend code for the user interface.
* `data/`: (Not tracked by Git) Will contain the cleaned local CSV files (e.g., Philadelphia restaurant data).
* `models/`: Core algorithmic implementations, including from-scratch K-Means and retrieval logic.
* `notebooks/`: Jupyter notebooks for data exploration, MVP testing, and baseline model verification.
* `scripts/`: Shell scripts for environment setup and activation.
* `requirements.txt`: Project dependencies (managed via `uv`).

## 📥 Data Setup

Due to the massive size of the Yelp Open Dataset, raw and cleaned data files are **not** tracked in this GitHub repository. To run this project, you must manually download the required CSV files.

**1. Download the Data:**
* Fetch the cleaned dataset (e.g., `cleaned_philly_restaurants.csv`) from our team's shared Google Drive: [`https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing`](https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing)

**2. Place it in the Repo:**
* Move the downloaded files directly into the `data/` folder at the root of this project. 
* Your local directory structure should look exactly like this before running any scripts:

```text
commercial-dining-intelligence/
├── app/
├── data/
│   ├── .gitkeep
│   └── cleaned_philly_restaurants.csv   <-- (Place your data here!)
├── models/
├── notebooks/
├── scripts/
├── .gitignore
├── README.md
└── requirements.txt
```
*Note: The `data/` directory is intentionally ignored by `.gitignore` to prevent large files from crashing the repository. Please do not force-add data files to Git.*

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
