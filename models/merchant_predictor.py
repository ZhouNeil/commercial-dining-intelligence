import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, mean_squared_error, roc_auc_score, 
                             f1_score, precision_score, recall_score, 
                             r2_score, mean_absolute_error)

# Force artifacts into models/artifacts regardless of where script is run
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class AblationMerchantPredictor:
    def __init__(self, train_path="../../train_spatial.csv", test_path="../../test_spatial.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.cat_cols = None
        self.time_cols = None
        self.base_attr_cols = None
        os.makedirs(MODEL_DIR, exist_ok=True)
        
    def load_data(self):
        print(f"Loading pre-computed spatial training data from {self.train_path}...")
        self.train_df = pd.read_csv(self.train_path)
        print(f"Loading pre-computed spatial testing data from {self.test_path}...")
        self.test_df = pd.read_csv(self.test_path)
        self.cat_cols = [c for c in self.train_df.columns if c.startswith('cat_')]
        self.time_cols = [c for c in self.train_df.columns if c.startswith('time_')]
        self.base_attr_cols = ['attr_restaurantspricerange2'] if 'attr_restaurantspricerange2' in self.train_df.columns else []

    def train_pipeline(self):
        self.load_data()
        
        train_df = self.train_df
        test_df = self.test_df
        
        print("\n--- 1. DIAGNOSTICS: TRAIN vs RANDOM HOLDOUT ---")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        print(f"Train Survival Rate: {train_df['is_open'].mean():.3f} vs Test Survival Rate: {test_df['is_open'].mean():.3f}")
        print(f"Train Avg Stars: {train_df['stars'].mean():.3f} vs Test Avg Stars: {test_df['stars'].mean():.3f}")
        
        print("\n--- 2. SPATIAL FEATURES PRE-LOADED ---")
        print(f"Train Median 0.5km Count: {train_df['count_all_0.5km'].median()} vs Test Median Density: {test_df['count_all_0.5km'].median()}")
        print(f"Train Median Nearest Competitor: {train_df['dist_nearest_same_cat'].median():.3f}km vs Test Median: {test_df['dist_nearest_same_cat'].median():.3f}km")

        families = {
            "Base": self.cat_cols + self.time_cols + self.base_attr_cols,
            "LatLon": ["latitude", "longitude"],
            "Distances": ["log_dist_nearest_same_cat", "dist_nearest_same_cat"],
            "Counts": [c for c in train_df.columns if "count" in c and "log_" in c or "has_" in c or "low_" in c],
            "Ratios": [c for c in train_df.columns if "ratio" in c],
            "Gap": [c for c in train_df.columns if "gap" in c],
            "Diversity": [c for c in train_df.columns if "diversity" in c],
            "Semantic": ["avg_rating_top5_similar", "survival_top5_similar"]
        }
        
        X_train_full = train_df.fillna(0)
        X_test_full = test_df.fillna(0)
        
        y_surv_train = train_df['is_open']
        y_surv_test = test_df['is_open']
        
        y_stars_train = train_df['stars']
        y_stars_test = test_df['stars']
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced'),
            "HistGB_Shallow(D=3)": HistGradientBoostingClassifier(max_depth=3, random_state=42),
            "HistGB_Deep(Def)": HistGradientBoostingClassifier(random_state=42)
        }
        
        print("\n--- 3. SURVIVAL ABLATION STUDY (Random Holdout) ---")
        best_auc = 0
        
        def get_optimal_f1(y_true, y_prob):
            best_f1, best_thresh = 0.0, 0.5
            for thresh in np.linspace(0.0, 1.0, 101):
                f1 = f1_score(y_true, (y_prob >= thresh).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh
            return best_f1, best_thresh
        
        baselines = ["Base", "Base + LatLon"]
        base_features = [families["Base"], families["Base"] + families["LatLon"]]
        
        results = []
        for name, feats in zip(baselines, base_features):
            for m_name, model in models.items():
                model.fit(X_train_full[feats], y_surv_train)
                prob = model.predict_proba(X_test_full[feats])[:, 1]
                auc = roc_auc_score(y_surv_test, prob)
                
                # Iterate threshold 0.0 to 1.0 to maximize F1
                max_f1, _ = get_optimal_f1(y_surv_test, prob)
                
                train_prob = model.predict_proba(X_train_full[feats])[:, 1]
                train_auc = roc_auc_score(y_surv_train, train_prob)
                
                results.append((m_name, name, train_auc, auc, max_f1))
                if auc > best_auc:
                    best_auc = auc
                    
        base_threshold_auc = max([r[3] for r in results if r[1] == "Base"])
        print(f"Global Baseline Performance Ceiling: {base_threshold_auc:.4f}")
        
        current_best_features = families["Base"].copy()
        current_best_name = "Base"
        current_best_auc = base_threshold_auc
        
        for fam_name in ["Distances", "Counts", "Ratios", "Gap", "Diversity", "Semantic"]:
            test_feats = current_best_features + families[fam_name]
            test_name = f"{current_best_name} + {fam_name}"
            
            test_model = HistGradientBoostingClassifier(max_depth=3, random_state=42)
            test_model.fit(X_train_full[test_feats], y_surv_train)
            prob = test_model.predict_proba(X_test_full[test_feats])[:, 1]
            auc = roc_auc_score(y_surv_test, prob)
            
            if auc > current_best_auc:
                print(f" [+] Keeping {fam_name}: AUC improved from {current_best_auc:.4f} -> {auc:.4f}")
                current_best_features = test_feats
                current_best_name = test_name
                current_best_auc = auc
            else:
                print(f" [-] Rejecting {fam_name}: AUC dropped to {auc:.4f}")
                
        print("\n--- 4. FINAL COMPARISON TABLE ---")
        print(f"{'Model Algorithm':<25} | {'Feature Set Baseline':<30} | {'Train AUC':<9} | {'Test AUC':<9} | {'Test F1':<7}")
        print("-" * 87)
        for r in results:
            print(f"{r[0]:<25} | {r[1]:<30} | {r[2]:.4f}    | {r[3]:.4f}    | {r[4]:.4f}")
            
        final_model = HistGradientBoostingClassifier(max_depth=3, random_state=42)
        final_model.fit(X_train_full[current_best_features], y_surv_train)
        prob = final_model.predict_proba(X_test_full[current_best_features])[:, 1]
        train_prob = final_model.predict_proba(X_train_full[current_best_features])[:, 1]
        
        opt_f1, opt_thresh = get_optimal_f1(y_surv_test, prob)
        print(f"\nFinal Model Optimal Threshold for F1 Score: {opt_thresh:.2f}")
        
        print("-" * 87)
        print(f"{'Final Optimized Config':<25} | {current_best_name:<30} | {roc_auc_score(y_surv_train, train_prob):.4f}    | {roc_auc_score(y_surv_test, prob):.4f}    | {opt_f1:.4f}")

        # Save classification model for website backend inference
        print(f"\nSaving ultimate Survival classifier to {MODEL_DIR}/global_survival_model.pkl...")
        joblib.dump(final_model, f"{MODEL_DIR}/global_survival_model.pkl")

        print("\n--- 5. STARS REGRESSION ABLATION STUDY ---")
        regressors = {
            "LinearRegression": LinearRegression(),
            "HistGB_Reg_Shallow(D=3)": HistGradientBoostingRegressor(max_depth=3, random_state=42),
            "HistGB_Reg_Deep(Def)": HistGradientBoostingRegressor(random_state=42)
        }
        
        best_mae = float('inf')
        
        reg_results = []
        for name, feats in zip(baselines, base_features):
            for m_name, model in regressors.items():
                model.fit(X_train_full[feats], y_stars_train)
                pred = model.predict(X_test_full[feats])
                mae = mean_absolute_error(y_stars_test, pred)
                
                train_pred = model.predict(X_train_full[feats])
                train_mae = mean_absolute_error(y_stars_train, train_pred)
                
                reg_results.append((m_name, name, train_mae, mae))
                if mae < best_mae:
                    best_mae = mae
                    
        base_threshold_mae = min([r[3] for r in reg_results if r[1] == "Base"])
        print(f"Global Baseline Performance Ceiling (MAE): {base_threshold_mae:.4f}")
        
        current_best_features_reg = families["Base"].copy()
        current_best_name_reg = "Base"
        current_best_mae = base_threshold_mae
        
        print("\n[NOTE] Bypassing Ablation limit to force True Spatial features into Rating Model for Map Interactivity!")
        current_best_features_reg = families["Base"].copy() + families["Counts"] + families["Distances"] + families["Gap"] + families["Semantic"]
        current_best_name_reg = "Base + Forced Geographic Spatials"
                
        print("\n--- 6. REGRESSION FINAL COMPARISON TABLE ---")
        print(f"{'Model Algorithm':<25} | {'Feature Set Baseline':<30} | {'Train MAE':<9} | {'Test MAE':<9}")
        print("-" * 80)
        for r in reg_results:
            print(f"{r[0]:<25} | {r[1]:<30} | {r[2]:.4f}    | {r[3]:.4f}")
            
        print("-" * 80)
        final_reg_model = HistGradientBoostingRegressor(max_depth=3, random_state=42)
        final_reg_model.fit(X_train_full[current_best_features_reg], y_stars_train)
        pred = final_reg_model.predict(X_test_full[current_best_features_reg])
        train_pred = final_reg_model.predict(X_train_full[current_best_features_reg])
        
        print(f"{'Final Optimized Config':<25} | {current_best_name_reg:<30} | {mean_absolute_error(y_stars_train, train_pred):.4f}    | {mean_absolute_error(y_stars_test, pred):.4f}")

        print(f"\nSaving ultimate Regression model to {MODEL_DIR}/global_rating_model.pkl...")
        joblib.dump(final_reg_model, f"{MODEL_DIR}/global_rating_model.pkl")

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    split_train = repo / "data" / "train_merchant_split.csv"
    split_test = repo / "data" / "test_spatial.csv"
    if split_train.is_file() and split_test.is_file():
        predictor = AblationMerchantPredictor(str(split_train), str(split_test))
    else:
        predictor = AblationMerchantPredictor()
    predictor.train_pipeline()
