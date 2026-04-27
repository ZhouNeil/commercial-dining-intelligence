import functools
import requests
from typing import Optional, Tuple

@functools.lru_cache(maxsize=128)
def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """
    Geocodes an address to (latitude, longitude) using OpenStreetMap's Nominatim API.
    Results are cached in memory to avoid repeated network calls and rate limits.
    Returns (lat, lon) or None if not found or error.
    """
    if not address or not address.strip():
        return None
        
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address.strip(),
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "CommercialDiningIntelligence/1.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon
    except Exception:
        pass
        
    return None
