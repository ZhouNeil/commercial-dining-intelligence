"""
Dimensionality Reduction Pipeline (PCA)
Owner: Carolina (Role 5)
Requirement: Implement Chapter 6. Reduce high-dimensional sparse categorical 
features into dense latent continuous features (Business DNA).
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

def fit_transform_pca(X_train, n_components=20):
    """
    Fit the PCA model on the training data and transform it.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The high-dimensional sparse feature matrix.
    n_components (int): The number of principal components to keep.
    
    Returns:
    tuple: (pca_model, X_reduced)
           pca_model: The fitted sklearn PCA object (to be saved for later use).
           X_reduced: The transformed lower-dimensional matrix.
    """
    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # initialize and fit PCA
    pca_model = PCA(n_components=n_components, random_state=42)
    X_reduced = pca_model.fit_transform(X_scaled)
    
    return (scaler, pca_model), X_reduced

def save_pca_model(models, filepath):
    """
    Serialize and save the fitted PCA model to the saved_models/ directory.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(models, filepath)
    print(f"Model saved successfully to {filepath}")

def load_pca_model(filepath):
    """
    Load the fitted PCA model for online inference.
    """
    return joblib.load(filepath)

def transform_new_query(models, user_input_vector):
    """
    Transform a brand new user query (e.g., from the Streamlit UI) using the fitted PCA.
    
    Parameters:
    pca_model: The loaded PCA object.
    user_input_vector (np.ndarray): A 1D array representing the user's selected attributes.
    
    Returns:
    np.ndarray: The compressed latent feature vector.
    """
    scaler, pca_model = models
    
    if user_input_vector.ndim == 1:
        user_input_vector = user_input_vector.reshape(1, -1)
    
    # scaling
    X_scaled = scaler.transform(user_input_vector)
    
    # project into PCA space
    X_latent = pca_model.transform(X_scaled)
    
    return X_latent
