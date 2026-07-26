"""
category_configs.py
--------------------
One config per product category. Each config declares:
  - which CSV to load and which column is price / rating
  - the scoring "dims" (non-longevity dimensions) and which column feeds each
  - how longevity is computed for this category (varies a lot -- laptops have
    warranty+CPU generation, phones have no rating at all so we lean on
    recency instead, headphones have no warranty/battery data so we lean on
    rating + review count as a "proven in the market" proxy)
  - use cases: per-use-case weights (must include "longevity") and
    sufficiency targets for every non-longevity dim

This is intentionally config-driven rather than one-python-function-per-category
so the SAME scoring engine (engine.py) works for every category without
duplicating the MCDA math five times.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CATEGORY_CONFIGS = {

    # ------------------------------------------------------------------
    "laptop": {
        "label": "Laptops",
        "csv": "laptops_clean.csv",  # produced by the existing laptop clean_data.py pipeline
        "id_col": "laptop_id",
        "price_col": "price_inr",
        "dims": {
            "cpu":     {"label": "processor power"},
            "ram":     {"label": "RAM"},
            "storage": {"label": "storage"},
            "gpu":     {"label": "graphics performance"},
            "display": {"label": "display quality"},
        },
        # laptop longevity/dims already computed by the existing laptop scoring
        # code (score_pool in laptop_engine.py) -- see engine.py's special-case.
        "use_cases": {
            "office":   {"label": "Office & Productivity", "weights": {"cpu": .15, "ram": .20, "storage": .15, "gpu": 0, "display": .15, "longevity": .35}, "targets": {"cpu": 11, "ram": 16, "storage": 512, "gpu": 0, "display": 1.4}, "primary": ["longevity", "ram"]},
            "gaming":   {"label": "Gaming", "weights": {"cpu": .20, "ram": .15, "storage": .15, "gpu": .40, "display": .10, "longevity": 0}, "targets": {"cpu": 15, "ram": 16, "storage": 1024, "gpu": 11, "display": 2.1}, "primary": ["gpu", "cpu"]},
            "coding":   {"label": "Programming & Development", "weights": {"cpu": .30, "ram": .30, "storage": .20, "gpu": .05, "display": .05, "longevity": .10}, "targets": {"cpu": 15, "ram": 16, "storage": 512, "gpu": 3, "display": 1.4}, "primary": ["cpu", "ram"]},
            "creative": {"label": "Video / Photo Editing & Design", "weights": {"cpu": .25, "ram": .20, "storage": .20, "gpu": .20, "display": .15, "longevity": 0}, "targets": {"cpu": 15, "ram": 32, "storage": 1024, "gpu": 11, "display": 3.7}, "primary": ["gpu", "display"]},
            "student":  {"label": "Student", "weights": {"cpu": .15, "ram": .20, "storage": .15, "gpu": 0, "display": .15, "longevity": .35}, "targets": {"cpu": 8, "ram": 8, "storage": 256, "gpu": 0, "display": 1.4}, "primary": ["longevity", "ram"]},
        },
    },

    # ------------------------------------------------------------------
    "phone": {
        "label": "Phones",
        "csv": "phones_clean.csv",
        "id_col": "item_id",
        "price_col": "price_inr",
        "dims": {
            "ram":     {"label": "RAM", "col": "ram_gb"},
            "battery": {"label": "battery capacity", "col": "battery_mah"},
            "camera":  {"label": "camera resolution", "col": "back_camera_mp"},
            "screen":  {"label": "screen size", "col": "screen_size_in"},
        },
        # No rating column exists in this dataset at all -- longevity leans on
        # recency (newer launch year = will stay relevant longer) and RAM
        # headroom (a proxy for how long the phone stays capable).
        "longevity_formula": {"launched_year": 0.6, "ram_gb": 0.4},
        "use_cases": {
            "gaming":       {"label": "Gaming", "weights": {"ram": .35, "battery": .25, "camera": .05, "screen": .15, "longevity": .20}, "targets": {"ram": 8, "battery": 5000, "camera": 50, "screen": 6.5}, "primary": ["ram", "battery"]},
            "camera":       {"label": "Photography", "weights": {"ram": .15, "battery": .15, "camera": .45, "screen": .10, "longevity": .15}, "targets": {"ram": 6, "battery": 4500, "camera": 108, "screen": 6.5}, "primary": ["camera"]},
            "battery_life": {"label": "All-Day Battery", "weights": {"ram": .10, "battery": .45, "camera": .10, "screen": .10, "longevity": .25}, "targets": {"ram": 4, "battery": 6000, "camera": 48, "screen": 6.5}, "primary": ["battery"]},
            "budget":       {"label": "Budget-Friendly", "weights": {"ram": .20, "battery": .25, "camera": .15, "screen": .10, "longevity": .30}, "targets": {"ram": 4, "battery": 4500, "camera": 48, "screen": 6.1}, "primary": ["longevity"]},
            "business":     {"label": "Business & Productivity", "weights": {"ram": .25, "battery": .30, "camera": .10, "screen": .15, "longevity": .20}, "targets": {"ram": 8, "battery": 5000, "camera": 50, "screen": 6.5}, "primary": ["battery", "ram"]},
        },
    },

    # ------------------------------------------------------------------
    "headphone": {
        "label": "Headphones & Earbuds",
        "csv": "headphones_clean.csv",
        "id_col": "item_id",
        "price_col": "price_inr",
        # This dataset has almost no hardware specs -- no battery life, no
        # ANC info, no warranty. Scoring leans mostly on rating, review
        # count (as a trust/popularity signal), and whether the physical
        # type matches what the use case actually needs.
        "dims": {
            "rating_dim":  {"label": "user rating", "col": "rating"},
            "popularity":  {"label": "review count", "col": "num_ratings"},
        },
        "longevity_formula": {"rating": 0.7, "num_ratings": 0.3},
        "type_preference": {
            # use_case -> preferred `type` value(s), used as a bonus/label in reasoning
            "gym_workout": ["True Wireless"],
            "travel": ["On the Ear", "True Wireless"],
            "studio_critical": ["On the Ear"],
            "budget": [],
        },
        "use_cases": {
            "gym_workout":     {"label": "Gym & Workouts", "weights": {"rating_dim": .5, "popularity": .3, "longevity": .2}, "targets": {"rating_dim": 4.3, "popularity": 5000}, "primary": ["rating_dim"]},
            "travel":          {"label": "Travel & Commute", "weights": {"rating_dim": .45, "popularity": .35, "longevity": .2}, "targets": {"rating_dim": 4.2, "popularity": 3000}, "primary": ["rating_dim"]},
            "studio_critical": {"label": "Critical Listening", "weights": {"rating_dim": .55, "popularity": .25, "longevity": .2}, "targets": {"rating_dim": 4.4, "popularity": 2000}, "primary": ["rating_dim"]},
            "budget":          {"label": "Budget-Friendly", "weights": {"rating_dim": .4, "popularity": .3, "longevity": .3}, "targets": {"rating_dim": 3.9, "popularity": 1000}, "primary": ["rating_dim"]},
        },
    },

    # ------------------------------------------------------------------
    "watch": {
        "label": "Smartwatches & Fitness Trackers",
        "csv": "watches_clean.csv",
        "id_col": "item_id",
        "price_col": "price_inr",
        "dims": {
            "battery_days": {"label": "battery life", "col": "battery_life_days"},
        },
        "longevity_formula": {"rating": 0.6, "battery_life_days": 0.4},
        "device_type_preference": {
            "fitness_tracking": "FitnessBand",
            "fashion_smartwatch": "Smartwatch",
        },
        "use_cases": {
            "fitness_tracking":   {"label": "Fitness Tracking", "weights": {"battery_days": .5, "longevity": .5}, "targets": {"battery_days": 10}, "primary": ["battery_days"]},
            "fashion_smartwatch": {"label": "Everyday Smartwatch", "weights": {"battery_days": .35, "longevity": .65}, "targets": {"battery_days": 5}, "primary": ["longevity"]},
            "budget":             {"label": "Budget-Friendly", "weights": {"battery_days": .4, "longevity": .6}, "targets": {"battery_days": 7}, "primary": ["longevity"]},
        },
    },

    # ------------------------------------------------------------------
    "tablet": {
        "label": "Tablets",
        "csv": "tablets_clean.csv",
        "id_col": "item_id",
        "price_col": "price_inr",
        "dims": {
            "ram":     {"label": "RAM", "col": "ram_gb"},
            "storage": {"label": "storage", "col": "storage_gb"},
            "battery": {"label": "battery capacity", "col": "battery_mah"},
            "display": {"label": "display quality", "col": "display_size_in"},
        },
        "longevity_formula": {"rating": 0.7, "ram_gb": 0.3},
        "use_cases": {
            "student":      {"label": "Student", "weights": {"ram": .20, "storage": .15, "battery": .25, "display": .15, "longevity": .25}, "targets": {"ram": 4, "storage": 64, "battery": 6000, "display": 10}, "primary": ["battery", "longevity"]},
            "creative":     {"label": "Drawing & Creative Work", "weights": {"ram": .30, "storage": .20, "battery": .15, "display": .25, "longevity": .10}, "targets": {"ram": 8, "storage": 128, "battery": 7000, "display": 11}, "primary": ["ram", "display"]},
            "entertainment": {"label": "Streaming & Entertainment", "weights": {"ram": .15, "storage": .20, "battery": .30, "display": .30, "longevity": .05}, "targets": {"ram": 4, "storage": 64, "battery": 7000, "display": 10.5}, "primary": ["display", "battery"]},
            "business":     {"label": "Business & Productivity", "weights": {"ram": .25, "storage": .20, "battery": .25, "display": .10, "longevity": .20}, "targets": {"ram": 8, "storage": 128, "battery": 7000, "display": 10}, "primary": ["ram", "battery"]},
        },
    },
}
