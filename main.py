"""
main.py - Multi-Category Recommendation Backend
--------------------------------------------------
Run with: python main.py  (or streamlit/uvicorn as before)

Same MCDA scoring philosophy as the original laptop-only backend, now
generalized across 5 categories: laptop, phone, headphone, watch, tablet.
See category_configs.py for what specs/use-cases each category has, and
engine.py for the shared scoring math.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from category_configs import CATEGORY_CONFIGS, DATA_DIR
from engine import score_and_rank, build_reasoning

# ============================================================
# LOAD ALL CATEGORY DATA AT STARTUP
# ============================================================
CATEGORY_DATA: dict[str, pd.DataFrame] = {}
for cat, config in CATEGORY_CONFIGS.items():
    path = os.path.join(DATA_DIR, config["csv"])
    df = pd.read_csv(path)
    CATEGORY_DATA[cat] = df
    print(f"[startup] Loaded {len(df)} items for category '{cat}'")


def _clean_for_json(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and np.isnan(v):
            cleaned[k] = None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            cleaned[k] = bool(v)
        else:
            cleaned[k] = v
    return cleaned


# ============================================================
# SCHEMAS
# ============================================================
class FilterParams(BaseModel):
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    brand: Optional[str] = None
    search: Optional[str] = None


class RecommendRequest(BaseModel):
    category: str = Field(..., description="laptop | phone | headphone | watch | tablet")
    filters: Optional[FilterParams] = None
    use_case: str
    top_n: int = 12


class CompareRequest(BaseModel):
    category: str
    item_ids: List[int] = Field(..., min_length=1, max_length=3)
    use_case: Optional[str] = None


def apply_filters(df: pd.DataFrame, category: str, filters: Optional[FilterParams]) -> pd.DataFrame:
    config = CATEGORY_CONFIGS[category]
    price_col = config["price_col"]
    out = df
    if not filters:
        return out
    if filters.min_price is not None:
        out = out[out[price_col] >= filters.min_price]
    if filters.max_price is not None:
        out = out[out[price_col] <= filters.max_price]
    if filters.brand:
        out = out[out["brand"].str.lower() == filters.brand.lower()]
    if filters.search:
        q = filters.search.lower()
        out = out[out["model_name"].str.lower().str.contains(q, na=False) | out["brand"].str.lower().str.contains(q, na=False)]
    return out


def _validate_category(category: str):
    if category not in CATEGORY_CONFIGS:
        raise HTTPException(status_code=400, detail=f"category must be one of {list(CATEGORY_CONFIGS)}")


def _validate_use_case(category: str, use_case: str):
    valid = CATEGORY_CONFIGS[category]["use_cases"]
    if use_case not in valid:
        raise HTTPException(status_code=400, detail=f"use_case for '{category}' must be one of {list(valid)}")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Multi-Category Recommendation API")
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_private_network=True, )


@app.get("/")
def root():
    return {
        "message": "Multi-category recommendation API is running. See /docs.",
        "categories": {cat: len(df) for cat, df in CATEGORY_DATA.items()},
    }


@app.get("/categories")
def list_categories():
    return {
        cat: {"label": config["label"], "item_count": len(CATEGORY_DATA[cat])}
        for cat, config in CATEGORY_CONFIGS.items()
    }


@app.get("/use-cases")
def get_use_cases(category: str):
    _validate_category(category)
    return {
        key: {"label": uc["label"]}
        for key, uc in CATEGORY_CONFIGS[category]["use_cases"].items()
    }


@app.get("/filter-options")
def get_filter_options(category: str):
    _validate_category(category)
    df = CATEGORY_DATA[category]
    price_col = CATEGORY_CONFIGS[category]["price_col"]
    return {
        "brands": sorted(df["brand"].dropna().unique().tolist()),
        "price_min": int(df[price_col].min()),
        "price_max": int(df[price_col].max()),
    }


@app.get("/items")
def list_items(
    category: str,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    brand: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    _validate_category(category)
    filters = FilterParams(min_price=min_price, max_price=max_price, brand=brand, search=search)
    result = apply_filters(CATEGORY_DATA[category], category, filters).iloc[offset: offset + limit]
    return [_clean_for_json(r) for r in result.to_dict(orient="records")]


@app.get("/items/{category}/{item_id}")
def get_item(category: str, item_id: int):
    _validate_category(category)
    config = CATEGORY_CONFIGS[category]
    df = CATEGORY_DATA[category]
    match = df[df[config["id_col"]] == item_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Item not found")
    return _clean_for_json(match.iloc[0].to_dict())


@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    _validate_category(req.category)
    _validate_use_case(req.category, req.use_case)

    candidates = apply_filters(CATEGORY_DATA[req.category], req.category, req.filters).reset_index(drop=True)
    if len(candidates) == 0:
        return []
    if len(candidates) < 2:
        row = candidates.iloc[0]
        return [{"item": _clean_for_json(row.to_dict()), "final_score": 1.0, "score_breakdown": {}, "reasoning": {}}]

    ranked = score_and_rank(candidates, req.category, req.use_case)
    top = ranked.head(req.top_n)
    results = []
    for _, row in top.iterrows():
        reasoning = build_reasoning(row, ranked, req.category, req.use_case)
        drop_cols = [c for c in row.index if c.startswith("dim_")]
        results.append({
            "item": _clean_for_json(row.drop(labels=drop_cols).to_dict()),
            "final_score": round(float(row["final_score"]), 4),
            "score_breakdown": {
                "performance_fit": round(float(row["performance_fit"]), 4),
                "value_score": round(float(row["value_score"]), 4),
                "longevity_score": round(float(row["longevity_score"]), 4),
            },
            "reasoning": reasoning,
        })
    return results


@app.post("/compare")
def compare_items(req: CompareRequest):
    _validate_category(req.category)
    config = CATEGORY_CONFIGS[req.category]
    df = CATEGORY_DATA[req.category]
    rows = df[df[config["id_col"]].isin(req.item_ids)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="None of the given item_ids were found")

    if req.use_case:
        _validate_use_case(req.category, req.use_case)
        scored = score_and_rank(rows, req.category, req.use_case)
        out = []
        for _, row in scored.iterrows():
            drop_cols = [c for c in row.index if c.startswith("dim_")]
            out.append({
                "item": _clean_for_json(row.drop(labels=drop_cols).to_dict()),
                "final_score": round(float(row["final_score"]), 4),
                "score_breakdown": {
                    "performance_fit": round(float(row["performance_fit"]), 4),
                    "value_score": round(float(row["value_score"]), 4),
                    "longevity_score": round(float(row["longevity_score"]), 4),
                },
                "reasoning": build_reasoning(row, scored, req.category, req.use_case),
            })
        out.sort(key=lambda r: -r["final_score"])
        return out
    return [{"item": _clean_for_json(r)} for r in rows.to_dict(orient="records")]


if __name__ == "__main__":
    import uvicorn
    print("\nStarting multi-category server...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
