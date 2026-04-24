"""
Backward-compatible re-export.

The recommendation retrieval logic lives in `backend/dining_retrieval/core/retrieval.py`; this module re-exports
for any legacy `from models.retrieval import ...` imports.
"""

from dining_retrieval.core.retrieval import RestaurantSearchIndex, TouristRetrieval

__all__ = ["RestaurantSearchIndex", "TouristRetrieval"]
