"""
Step 5 of the system design: persist structured predictions to a product
catalog. This prototype uses a local JSON file to stand in for a real
database (e.g., Postgres/DynamoDB) in a production system.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "product_catalog.json")


def save_to_catalog(
    image_filename: str, category: str, confidence: float, description: str
) -> Dict:
    """Append one structured prediction to the catalog file and return it."""
    entry = {
        "image_filename": image_filename,
        "category": category,
        "category_confidence": round(confidence, 4),
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    catalog = load_catalog()
    catalog.append(entry)

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    return entry


def load_catalog() -> List[Dict]:
    """Load all saved catalog entries, or an empty list if none exist yet."""
    if not os.path.exists(CATALOG_PATH):
        return []
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)
