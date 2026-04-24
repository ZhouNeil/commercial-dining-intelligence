# Inference Pipeline Tests

This directory contains test scripts that simulate how the frontend application behaves when interacting with our machine learning models.

## How to Run the Inference Test

The `test_inference.py` script mimics a live scenario where a user drops a map pin and selects a restaurant category. It instantly computes spatial features, queries the Custom KNN, and returns predictions using the saved global models.

### Prerequisites
Because `test_inference.py` relies on the globally generated datasets and serialized `.pkl` models (which are stored locally and ignored by Git due to their size), you must ensure you have the following files locally on your machine before running the test:
1. `train_spatial.csv` located in the parent directory of this repository (i.e. `../train_spatial.csv`)
2. `advanced_survival_classifier.pkl` located in `models/artifacts/`
3. `global_rating_model.pkl` located in `models/artifacts/`

### Running the Test
To ensure all file paths resolve correctly, you **must run the test from the root directory of the repository**, not from inside the `tests/` folder.

1. Open your terminal.
2. Navigate to the root of the repository (`commercial-dining-intelligence`).
3. Run the following command:

```bash
PYTHONPATH=backend:. python tests/test_inference.py
```

### Expected Output
If your environment is set up correctly, you should see output simulating a live frontend request in Philadelphia, returning live metrics, survival probability, and a predicted star rating.
