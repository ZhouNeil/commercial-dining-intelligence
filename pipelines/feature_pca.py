"""
Dimensionality Reduction Pipeline (PCA)
Owner: Carolina (Role 5)
Requirement: Implement Chapter 6. Reduce high-dimensional sparse categorical 
features into dense latent continuous features (Business DNA).
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

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
    # TODO for Carolina: Initialize PCA, fit_transform the data.
    # Tip: Don't forget to check if data scaling (StandardScaler) is needed before PCA!
    raise NotImplementedError("Carolina needs to implement the PCA fitting logic here.")

def save_pca_model(pca_model, filepath):
    """
    Serialize and save the fitted PCA model to the saved_models/ directory.
    """
    # TODO for Carolina: Use joblib.dump to save the model.
    pass

def load_pca_model(filepath):
    """
    Load the fitted PCA model for online inference.
    """
    # TODO for Carolina: Use joblib.load to read the model.
    pass

def transform_new_query(pca_model, user_input_vector):
    """
    Transform a brand new user query (e.g., from the web UI) using the fitted PCA.
    
    Parameters:
    pca_model: The loaded PCA object.
    user_input_vector (np.ndarray): A 1D array representing the user's selected attributes.
    
    Returns:
    np.ndarray: The compressed latent feature vector.
    """
    # TODO for Carolina: Use pca_model.transform() on the new input.
    pass
