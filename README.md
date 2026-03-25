# Yelp Dual-Mode Commercial & Dining Intelligence App

This repository contains the codebase for our Machine Learning course final project. The application utilizes the Yelp Open Dataset to provide insights for both prospective merchants and tourists.

## 📂 Repository Structure

* `app/`: Streamlit frontend code for the user interface.
* `data/`: (Not tracked by Git) Will contain the cleaned local CSV files (e.g., Philadelphia restaurant data).
* `models/`: Core algorithmic implementations, including from-scratch K-Means and retrieval logic.
* `notebooks/`: Jupyter notebooks for data exploration, MVP testing, and baseline model verification.
* `scripts/`: Shell scripts for environment setup and activation.
* `requirements.txt`: Project dependencies (managed via `uv`).

## 🚀 Setup Environment

We use `uv` for lightning-fast dependency management. To run this project locally, please follow these steps from the root directory of the project:

**1. Initial Setup (Run Once):**
This script will install `uv` (if necessary), create a virtual environment (`.venv`), and install all dependencies from `requirements.txt`.
```bash
source scripts/setup_env.sh

**2. Activate Environment (Run every time you code):**
Whenever you open a new terminal session to work on this project, activate the environment by running:
```bash
source scripts/activate_env.sh
