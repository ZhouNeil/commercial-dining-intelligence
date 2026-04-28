from __future__ import annotations

from typing import Optional


def normalize_state(state: Optional[str]) -> Optional[str]:
    """Normalize a state string to uppercase, returning None for blank/None input."""
    if state and str(state).strip():
        return str(state).strip().upper()
    return None
