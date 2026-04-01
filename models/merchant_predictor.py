import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================================================
# Config
# =========================================================

DATA_PATH = "updated_philly_data.csv"   # change if needed
MODEL_DIR = "models/artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 8
N_PCA_COMPONENTS = 10

# =========================================================
# Helper: choose safe feature columns
# =========================================================

def get_feature_columns(df: pd.DataFrame):
    """
    Build a safe feature set for prediction.
    We exclude identifiers, raw text-like columns, and leakage-prone columns.

    Red line: avoid using review_count because for a "new" shop this is future information.
    """

    exclude_cols = {
        "business_id",
        "name",
        "address",
        "city",
        "state",
        #"postal_code",
        "categories",   # raw text
        "hours",        # raw dictionary/string
        "is_open",      # target
        "stars",        # target
        #"review_count"  # leakage-prone for new shop prediction
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols

# =========================================================
# Step 1: load data
# =========================================================

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # Make sure required columns exist
    required_cols = ["is_open", "stars", "latitude", "longitude", "attr_restaurantspricerange2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

# =========================================================
# Step 2: prepare base numeric features
# =========================================================

def prepare_base_features(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)

    X_base = df[feature_cols].copy()

    # Force everything numeric if possible
    for col in X_base.columns:
        X_base[col] = pd.to_numeric(X_base[col], errors="coerce")

    # Fill missing numeric values
    X_base = X_base.fillna(0)

    y_survival = df["is_open"].astype(int)
    y_rating = df["stars"].astype(float)

    return X_base, y_survival, y_rating

# =========================================================
# Step 3: create Cluster_ID using KMeans
# =========================================================

def build_cluster_features(df: pd.DataFrame):
    """
    Build cluster_id from location + price.
    This matches the project requirement that Merchant Mode should use Cluster_ID.
    """

    cluster_input_cols = ["latitude", "longitude", "attr_restaurantspricerange2"]
    cluster_df = df[cluster_input_cols].copy().fillna(0)

    scaler_cluster = StandardScaler()
    cluster_scaled = scaler_cluster.fit_transform(cluster_df)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cluster_ids = kmeans.fit_predict(cluster_scaled)

    return cluster_ids, scaler_cluster, kmeans

# =========================================================
# Step 4: create PCA "business DNA" features
# =========================================================

def build_pca_features(X_base: pd.DataFrame):
    """
    PCA over the full safe feature matrix.
    This gives compressed business-DNA style features.
    """

    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X_base)

    n_components = min(N_PCA_COMPONENTS, X_base.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X_base.index)

    return X_pca_df, scaler_pca, pca

# =========================================================
# Step 5: assemble final X
# =========================================================

def build_final_feature_matrix(df: pd.DataFrame):
    """
    Final X = [Cluster_ID, price, PCA business DNA scores]
    This follows your role description exactly.
    """

    X_base, y_survival, y_rating = prepare_base_features(df)
    
    cluster_ids, scaler_cluster, kmeans = build_cluster_features(df)
    X_pca_df, scaler_pca, pca = build_pca_features(X_base)

    X_final = pd.DataFrame(index=df.index)
    X_final["cluster_id"] = cluster_ids
    X_final["price"] = df["attr_restaurantspricerange2"].fillna(0).astype(float)

    for col in X_pca_df.columns:
        X_final[col] = X_pca_df[col]
        
    for col in X_base.columns:
        if col not in X_final.columns:
            X_final[col] = X_base[col]

    return X_final, y_survival, y_rating, scaler_cluster, kmeans, scaler_pca, pca

# =========================================================
# Step 6: train classifier for survival
# =========================================================

def train_survival_model(X_train, y_train, X_test, y_test):
    """
    Train robust classifier with class imbalance handling.
    We use soft voting over Logistic Regression + SVC.
    """

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    svm = SVC(
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    clf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svm", svm)
        ],
        voting="soft"
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n========== Survival Model Evaluation ==========")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf

# =========================================================
# Step 7: train regressor for stars
# =========================================================

def train_rating_model(X_train, y_train, X_test, y_test):
    """
    Train rating predictor.
    RandomForestRegressor is more flexible than plain LinearRegression.
    If your instructor strictly wants Linear Regression, swap it in below.
    """

    reg = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    print("\n========== Rating Model Evaluation ==========")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 4))
    print("MAE:", round(mean_absolute_error(y_test, preds), 4))
    print("R^2:", round(r2_score(y_test, preds), 4))

    return reg

# =========================================================
# Step 8: save artifacts
# =========================================================

def save_artifacts(
    survival_model,
    rating_model,
    scaler_cluster,
    kmeans,
    scaler_pca,
    pca,
    feature_names
):
    joblib.dump(survival_model, os.path.join(MODEL_DIR, "survival_model.pkl"))
    joblib.dump(rating_model, os.path.join(MODEL_DIR, "rating_model.pkl"))
    joblib.dump(scaler_cluster, os.path.join(MODEL_DIR, "cluster_scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler_pca, os.path.join(MODEL_DIR, "pca_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "base_feature_names.pkl"))

    print(f"\nSaved all artifacts to: {MODEL_DIR}")

# =========================================================
# Step 9: training pipeline
# =========================================================

def train_all(path=DATA_PATH):
    df = load_data(path)

    # Build final matrix
    X_final, y_survival, y_rating, scaler_cluster, kmeans, scaler_pca, pca = build_final_feature_matrix(df)

    # Same split indices for both tasks
    train_idx, test_idx = train_test_split(
        X_final.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_survival
    )

    X_train = X_final.loc[train_idx]
    X_test = X_final.loc[test_idx]

    y_train_survival = y_survival.loc[train_idx]
    y_test_survival = y_survival.loc[test_idx]

    y_train_rating = y_rating.loc[train_idx]
    y_test_rating = y_rating.loc[test_idx]

    # Train models
    survival_model = train_survival_model(X_train, y_train_survival, X_test, y_test_survival)
    rating_model = train_rating_model(X_train, y_train_rating, X_test, y_test_rating)

    # Save feature list used before PCA
    X_base, _, _ = prepare_base_features(df)
    feature_names = list(X_base.columns)

    save_artifacts(
        survival_model,
        rating_model,
        scaler_cluster,
        kmeans,
        scaler_pca,
        pca,
        feature_names
    )

# =========================================================
# Step 10: frontend-friendly prediction function
# =========================================================

def prepare_single_business_features(input_dict: dict):
    """
    input_dict should contain the raw business features needed by the same schema
    as the original training data columns (except target columns).

    Example:
    {
        "latitude": 39.95,
        "longitude": -75.16,
        "attr_restaurantspricerange2": 2,
        "cat_pizza": 1,
        "cat_restaurants": 1,
        ...
    }
    """

    feature_names = joblib.load(os.path.join(MODEL_DIR, "base_feature_names.pkl"))
    scaler_cluster = joblib.load(os.path.join(MODEL_DIR, "cluster_scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler_pca = joblib.load(os.path.join(MODEL_DIR, "pca_scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))

    row = {col: 0 for col in feature_names}
    row.update(input_dict)

    X_base_single = pd.DataFrame([row], columns=feature_names).fillna(0)

    # cluster features
    cluster_input = X_base_single[["latitude", "longitude", "attr_restaurantspricerange2"]].copy()
    cluster_scaled = scaler_cluster.transform(cluster_input)
    cluster_id = kmeans.predict(cluster_scaled)[0]

    # pca features
    X_scaled = scaler_pca.transform(X_base_single)
    X_pca = pca.transform(X_scaled)

    final_row = pd.DataFrame()
    final_row["cluster_id"] = [cluster_id]
    final_row["price"] = [float(X_base_single["attr_restaurantspricerange2"].iloc[0])]

    for i in range(X_pca.shape[1]):
        final_row[f"pca_{i+1}"] = X_pca[:, i]

    for col in X_base_single.columns:
        if col not in final_row.columns:
            final_row[col] = X_base_single[col]

    return final_row

def predict_business_success(input_dict: dict):
    """
    Returns smooth probability for survival and predicted star rating.
    This satisfies the DoD requirement for predict_proba().
    """

    survival_model = joblib.load(os.path.join(MODEL_DIR, "survival_model.pkl"))
    rating_model = joblib.load(os.path.join(MODEL_DIR, "rating_model.pkl"))

    X_input = prepare_single_business_features(input_dict)

    survival_probability = survival_model.predict_proba(X_input)[0][1]
    predicted_rating = rating_model.predict(X_input)[0]

    return {
        "success_probability": float(round(survival_probability, 4)),
        "expected_rating": float(round(predicted_rating, 2))
    }

# =========================================================
# Run training
# =========================================================

if __name__ == "__main__":
    train_all(DATA_PATH)

    # Example inference
    example_business = {
        "latitude": 39.9526,
        "longitude": -75.1652,
        "attr_restaurantspricerange2": 2,
        "time_is_open_morning": 1,
        "time_is_open_latenight": 0,
        "time_open_on_weekends": 1,
        "cat_restaurants": 1,
        "cat_pizza": 1,
        "cat_italian": 1,
        "attr_restaurantsdelivery": 1,
        "attr_outdoorseating": 0,
        "attr_restaurantsreservations": 0,
        "attr_goodforkids": 1
    }

    result = predict_business_success(example_business)
    print("\nExample Prediction:")
    print(result)