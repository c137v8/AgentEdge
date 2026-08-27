"""
Mock product catalog.
In a real deployment this would hit a product/search service. Kept in-memory
and simple on purpose -- the interesting engineering in this project is the
mandate/guardrail layer, not the catalog.
"""
from typing import List, Dict, Optional

CATALOG: List[Dict] = [
    {"id": "sku_001", "name": "Wireless Earbuds Pro", "category": "electronics", "price": 2499, "stock": 14, "tags": ["audio", "bluetooth", "earbuds"]},
    {"id": "sku_002", "name": "Mechanical Keyboard 87-key", "category": "electronics", "price": 3999, "stock": 8, "tags": ["keyboard", "typing", "office"]},
    {"id": "sku_003", "name": "Stainless Steel Water Bottle 1L", "category": "home", "price": 599, "stock": 40, "tags": ["hydration", "steel", "eco"]},
    {"id": "sku_004", "name": "Running Shoes AeroFlex", "category": "fashion", "price": 3299, "stock": 20, "tags": ["shoes", "running", "sports"]},
    {"id": "sku_005", "name": "Cotton Bedsheet Set (Queen)", "category": "home", "price": 1799, "stock": 25, "tags": ["bedding", "cotton", "home"]},
    {"id": "sku_006", "name": "USB-C Fast Charger 65W", "category": "electronics", "price": 1299, "stock": 30, "tags": ["charger", "usb-c", "power"]},
    {"id": "sku_007", "name": "Yoga Mat Premium", "category": "fitness", "price": 899, "stock": 18, "tags": ["yoga", "fitness", "mat"]},
    {"id": "sku_008", "name": "Ceramic Coffee Mug Set (4)", "category": "home", "price": 649, "stock": 35, "tags": ["kitchen", "coffee", "ceramic"]},
    {"id": "sku_009", "name": "Backpack Urban 25L", "category": "fashion", "price": 2199, "stock": 12, "tags": ["bag", "travel", "college"]},
    {"id": "sku_010", "name": "Desk Lamp LED Adjustable", "category": "home", "price": 999, "stock": 22, "tags": ["lamp", "led", "desk"]},
]


def search(query: str, max_price: Optional[int] = None, category: Optional[str] = None) -> List[Dict]:
    q = (query or "").lower().strip()
    results = []
    for item in CATALOG:
        haystack = " ".join([item["name"].lower(), item["category"].lower(), *item["tags"]])
        if q and q not in haystack and not any(word in haystack for word in q.split()):
            continue
        if max_price is not None and item["price"] > max_price:
            continue
        if category and item["category"].lower() != category.lower():
            continue
        results.append(item)
    # if the query matched nothing, fall back to top items so the agent always has something to show
    return results or (CATALOG[:5] if not q else [])


def get_by_id(sku_id: str) -> Optional[Dict]:
    return next((i for i in CATALOG if i["id"] == sku_id), None)
