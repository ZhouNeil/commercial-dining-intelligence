"""
Backward-compatible re-export.

The recommendation retrieval logic now lives in `app/retrieval.py` so the Streamlit MVP
can keep all app-related code under one folder.
"""

from app.core.retrieval import RestaurantSearchIndex, TouristRetrieval

__all__ = ["RestaurantSearchIndex", "TouristRetrieval"]
