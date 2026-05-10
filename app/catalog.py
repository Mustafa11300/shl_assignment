"""
Catalog loader and lookup utilities for the SHL product catalog.

Loads 377 Individual Test Solutions from dataset.txt, normalizes fields,
and provides fast lookup by name, URL, and entity_id.
"""

import json
import os
from typing import Optional

# Test type full name → letter code mapping
TEST_TYPE_CODES = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Reverse mapping: letter code → full name
CODE_TO_TYPE = {v: k for k, v in TEST_TYPE_CODES.items()}


def _get_data_path() -> str:
    """Return the path to dataset.txt."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "catalog.json")


def _get_raw_data_path() -> str:
    """Return the path to the raw dataset.txt."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "dataset.txt")


def keys_to_test_type_code(keys: list[str]) -> str:
    """Convert list of test type names to comma-separated letter codes.
    
    Example: ["Knowledge & Skills", "Simulations"] → "K,S"
    """
    codes = []
    for key in keys:
        code = TEST_TYPE_CODES.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(sorted(codes))


def load_catalog() -> list[dict]:
    """Load the catalog from catalog.json (cleaned) or dataset.txt (raw).
    
    Returns a list of catalog items with normalized fields including
    a computed `test_type` field with letter codes.
    """
    catalog_path = _get_data_path()
    
    # If cleaned catalog exists, use it
    if os.path.exists(catalog_path):
        with open(catalog_path, "r") as f:
            return json.load(f)
    
    # Otherwise, load from raw dataset.txt and clean
    raw_path = _get_raw_data_path()
    with open(raw_path, "r") as f:
        data = json.load(f, strict=False)
    
    catalog = []
    for item in data:
        keys = item.get("keys", [])
        catalog.append({
            "entity_id": item.get("entity_id", ""),
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "description": item.get("description", ""),
            "test_type": keys_to_test_type_code(keys),
            "test_type_names": keys,
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        })
    
    # Save cleaned version
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    
    return catalog


class CatalogStore:
    """In-memory catalog store with fast lookup by name and URL."""
    
    def __init__(self):
        self.items = load_catalog()
        self._by_url = {item["url"]: item for item in self.items}
        self._by_name_lower = {item["name"].lower(): item for item in self.items}
        self._by_id = {item["entity_id"]: item for item in self.items}
    
    def get_by_url(self, url: str) -> Optional[dict]:
        """Look up a catalog item by its URL."""
        return self._by_url.get(url)
    
    def get_by_name(self, name: str) -> Optional[dict]:
        """Look up a catalog item by name (case-insensitive)."""
        return self._by_name_lower.get(name.lower())
    
    def url_exists(self, url: str) -> bool:
        """Check if a URL exists in the catalog."""
        return url in self._by_url
    
    def search_by_name(self, query: str) -> list[dict]:
        """Simple substring search by name."""
        query_lower = query.lower()
        return [item for item in self.items if query_lower in item["name"].lower()]
    
    def get_all_names(self) -> list[str]:
        """Return all product names."""
        return [item["name"] for item in self.items]
    
    def get_embedding_text(self, item: dict) -> str:
        """Build the text representation used for embedding a catalog item.
        
        Combines name, description, test types, and job levels into a single
        string optimized for semantic search.
        """
        parts = [
            item["name"],
            item.get("description", ""),
            f"Categories: {', '.join(item.get('test_type_names', []))}",
            f"Job levels: {', '.join(item.get('job_levels', []))}",
        ]
        if item.get("languages"):
            parts.append(f"Languages: {', '.join(item['languages'][:5])}")
        if item.get("duration"):
            parts.append(f"Duration: {item['duration']}")
        return ". ".join(parts)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]
