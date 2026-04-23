"""US state names ↔ USPS codes for query parsing and semantic filters."""

from __future__ import annotations

# Lowercase full / multi-word name -> two-letter code (incl. DC).
US_STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}

US_STATE_CODES: frozenset[str] = frozenset(US_STATE_NAME_TO_CODE.values())

# Longer phrases first so e.g. "new york" beats "york".
US_STATE_NAMES_ORDERED: tuple[str, ...] = tuple(
    sorted(US_STATE_NAME_TO_CODE.keys(), key=len, reverse=True)
)


def display_name_for_code(code: str) -> str:
    """Title-case state name for a USPS code (fallback: code itself)."""
    u = (code or "").strip().upper()
    for name, c in US_STATE_NAME_TO_CODE.items():
        if c == u:
            return " ".join(w.capitalize() for w in name.split())
    return u
