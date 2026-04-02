"""
Core k-NN Algorithm Implementation (From Scratch)
Owner: Yiwen (Role 2)
Requirement: Pure NumPy implementation. STRICTLY NO sklearn.neighbors.
"""
import numpy as np

class CustomKNN:
    def __init__(self):
        """
        Initialize the k-NN model.
        """
        self.X_train = None

    def fit(self, X):
        """
        Store the training data.
        
        Parameters:
        X (np.ndarray): The feature matrix of shape (n_samples, n_features).
        """
        self.X_train = np.array(X)
        # TODO for Yiwen: Add input validation here.

    def retrieve_by_radius(self, query_vector, radius, metric='euclidean'):
        """
        For Merchant Mode: Find all neighbors within a specific spatial radius.
        
        Parameters:
        query_vector (np.ndarray): The feature vector of the target location.
        radius (float): The maximum distance to be considered a neighbor.
        metric (str): Distance metric to use ('euclidean' or 'haversine').
        
        Returns:
        indices (list or np.ndarray): Indices of businesses within the radius.
        """
        # TODO for Yiwen: Implement pure NumPy broadcasting to calculate distances.
        # Ensure it runs quickly (no native Python for-loops for distance calculation).
        raise NotImplementedError("Yiwen needs to implement the radius search logic here.")

    def retrieve_top_k(self, query_vector, k=10, metric='cosine'):
        """
        For Tourist Mode: Find the top K most similar restaurants based on embeddings.
        
        Parameters:
        query_vector (np.ndarray): The semantic embedding of the user's query.
        k (int): Number of recommendations to return.
        metric (str): Similarity metric ('cosine' or 'euclidean').
        
        Returns:
        indices (list or np.ndarray): Indices of the top K recommendations.
        """
        # TODO for Yiwen: Implement cosine similarity using np.dot and np.linalg.norm.
        raise NotImplementedError("Yiwen needs to implement the top-k search logic here.")
