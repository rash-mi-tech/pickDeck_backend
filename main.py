"""
main.py - Laptop Recommendation Backend (Hackathon Edition)
-------------------------------------------------------------
Everything in one file on purpose: data cleaning, recommendation logic,
and the API server, so you can just hit F5 in IDLE (or run
`python main.py`) and it works. No separate setup steps.

HOW TO RUN
1. Put your raw dataset as "laptop.csv" in this same folder.
2. Install dependencies once (in IDLE: Run > this won't install for you,
   so do it from a terminal ONE time):
       pip install fastapi uvicorn pandas numpy scikit-learn pydantic
3. Open this file in IDLE and press F5 (or run `python main.py` from a
   terminal). You'll see "Uvicorn running on http://127.0.0.1:8000".
4. Open http://127.0.0.1:8000/docs in your browser to test it.

WHAT IT DOES
- Loads laptop.csv, parses messy spec text (RAM, SSD, CPU, GPU, etc.)
  into clean numeric/text fields.
- Serves a FastAPI backend with:
    GET  /laptops              filter/list laptops
    GET  /laptops/{id}         single laptop
    GET  /laptops/{id}/similar laptops similar to a given one
    POST /recommend            filters + use-case (gaming/student/etc)
- Recommendation = hard filters (budget, RAM, brand...) narrow the pool,
  then cosine similarity over a normalized spec vector ranks what's left.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# 1. DATA CLEANING  (raw CSV -> structured DataFrame, in memory)
# ============================================================

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptop.csv")

CPU_BRAND_PAT = re.compile(r"\b(Intel|AMD|Apple)\b", re.I)
GEN_PAT = re.compile(r"(\d+)(?:th|rd|nd|st)\s*Gen", re.I)
CPU_TIER_PAT = re.compile(
    r"\b(Core i[3579]|Ryzen [3579]|Ryzen AI [3579]|Celeron|Pentium|M1|M2|M3|M4|Athlon|Snapdragon)\b", re.I
)
CORES_PAT = re.compile(r"(\d+)\s*Core", re.I)
WORD_CORES_PAT = re.compile(r"\b(Single|Dual|Quad|Hexa|Octa|Deca)\s*Core", re.I)
WORD_TO_NUM = {"single": 1, "dual": 2, "quad": 4, "hexa": 6, "octa": 8, "deca": 10}
THREADS_PAT = re.compile(r"(\d+)\s*Threads", re.I)
RAM_PAT = re.compile(r"(\d+)\s*GB\s*((?:DDR|LPDDR)\d[a-zA-Z]*)?\s*RAM", re.I)
SSD_PAT = re.compile(r"(\d+)\s*(GB|TB)\s*SSD", re.I)
HDD_PAT = re.compile(r"(\d+)\s*(GB|TB)\s*Hard Disk", re.I)
DISPLAY_SIZE_PAT = re.compile(r"([\d.]+)\s*inches", re.I)
RESOLUTION_PAT = re.compile(r"(\d+)\s*x\s*(\d+)\s*pixels", re.I)
TOUCH_PAT = re.compile(r"Touch Screen", re.I)
GPU_VRAM_PAT = re.compile(r"(\d+)\s*GB\s*(NVIDIA|AMD|Intel)\b", re.I)
GPU_MODEL_PAT = re.compile(
    r"(RTX\s*\d{4}(?:\s*Ti)?|GTX\s*\d{4}(?:\s*Ti)?|Radeon RX\s*\w+|"
    r"Radeon Graphics|Iris Xe Graphics|UHD Graphics|Integrated Graphics|"
    r"M1 Pro|M1 Max|M1|M2 Pro|M2 Max|M2|M3 Pro|M3 Max|M3|M4 Pro|M4 Max|M4)", re.I
)
OS_PAT = re.compile(r"(Windows 11 Pro|Windows 11|Windows 10|Mac OS|DOS|Chrome OS)", re.I)
WARRANTY_PAT = re.compile(r"(\d+)\s*Year", re.I)
PRICE_PAT = re.compile(r"[\d,]+")
BRAND_LIST = [
    "HP", "Dell", "Lenovo", "Asus", "ASUS", "Acer", "MSI", "Apple", "Samsung",
    "LG", "Microsoft", "Infinix", "Tecno", "Realme", "Xiaomi", "Redmi",
    "Honor", "Gigabyte", "Fujitsu", "Vaio", "Chuwi", "Avita", "iBall",
]


def _extract(pattern, text, group=1, cast=str, default=None):
    m = pattern.search(text)
    if not m or m.group(group) is None:
        return default
    try:
        return cast(m.group(group))
    except (ValueError, IndexError, TypeError):
        return default


def _parse_row(row: pd.Series) -> dict:
    blob = " | ".join(str(v) for v in row.values if pd.notna(v))
    model_name = str(row.get("Model", ""))
    brand = next((b for b in BRAND_LIST if model_name.lower().startswith(b.lower())), "Other")

    price_match = PRICE_PAT.search(str(row.get("Price", "")))
    price = int(price_match.group(0).replace(",", "")) if price_match else None

    rating = row.get("Rating")
    rating = float(rating) if pd.notna(rating) else None

    cores = _extract(CORES_PAT, blob, cast=int)
    if cores is None:
        wm = WORD_CORES_PAT.search(blob)
        if wm:
            cores = WORD_TO_NUM.get(wm.group(1).lower())

    ssd_val = _extract(SSD_PAT, blob, group=1, cast=int)
    ssd_unit = _extract(SSD_PAT, blob, group=2, default="GB")
    ssd_gb = ssd_val * 1024 if (ssd_val and ssd_unit and ssd_unit.upper() == "TB") else (ssd_val or 0)

    hdd_val = _extract(HDD_PAT, blob, group=1, cast=int)
    hdd_unit = _extract(HDD_PAT, blob, group=2, default="GB")
    hdd_gb = hdd_val * 1024 if (hdd_val and hdd_unit and hdd_unit.upper() == "TB") else (hdd_val or 0)

    res_match = RESOLUTION_PAT.search(blob)
    if res_match:
        w, h = int(res_match.group(1)), int(res_match.group(2))
        resolution_px = max(w, h) * min(w, h)
        resolution_label = f"{w}x{h}"
    else:
        resolution_px, resolution_label = None, None

    gpu_model = _extract(GPU_MODEL_PAT, blob, default="Integrated")
    gpu_vram_match = GPU_VRAM_PAT.search(blob)
    gpu_vram_gb = int(gpu_vram_match.group(1)) if gpu_vram_match else 0
    discrete_names = ("RTX", "GTX", "Radeon RX")
    gpu_dedicated = gpu_vram_gb > 0 or (gpu_model and any(g in gpu_model.upper() for g in discrete_names))

    cpu_brand_raw = _extract(CPU_BRAND_PAT, blob)
    cpu_brand = {"intel": "Intel", "amd": "AMD", "apple": "Apple"}.get(
        (cpu_brand_raw or "").lower(), cpu_brand_raw
    )

    return {
        "model_name": model_name,
        "brand": brand,
        "price_inr": price,
        "rating": rating,
        "cpu_brand": cpu_brand,
        "cpu_tier": _extract(CPU_TIER_PAT, blob),
        "cpu_generation": _extract(GEN_PAT, blob, cast=int),
        "cpu_cores": cores,
        "cpu_threads": _extract(THREADS_PAT, blob, cast=int),
        "ram_gb": _extract(RAM_PAT, blob, group=1, cast=int),
        "ram_type": _extract(RAM_PAT, blob, group=2, default="DDR4"),
        "ssd_gb": ssd_gb,
        "hdd_gb": hdd_gb,
        "display_size_in": _extract(DISPLAY_SIZE_PAT, blob, cast=float),
        "resolution_label": resolution_label,
        "resolution_px": resolution_px,
        "touchscreen": bool(TOUCH_PAT.search(blob)),
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram_gb,
        "gpu_dedicated": bool(gpu_dedicated),
        "os": _extract(OS_PAT, blob, default="Unspecified"),
        "warranty_years": _extract(WARRANTY_PAT, blob, cast=int, default=1),
    }


def load_and_clean(csv_path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Couldn't find '{csv_path}'. Put your laptop.csv in the same folder as main.py."
        )
    df_raw = pd.read_csv(csv_path)
    parsed = df_raw.apply(_parse_row, axis=1, result_type="expand")
    parsed.insert(0, "laptop_id", range(1, len(parsed) + 1))
    before = len(parsed)
    parsed = parsed.dropna(subset=["price_inr", "ram_gb"]).reset_index(drop=True)
    print(f"[startup] Parsed {len(parsed)} laptops ({before - len(parsed)} dropped for missing price/RAM).")
    return parsed


LAPTOPS_DF = load_and_clean()


LAPTOPS_DF = load_and_clean()

# ============================================================
# 2. MULTI-CRITERIA DECISION ENGINE
# ============================================================
#
# Recommendations are NOT sorted by price, and NOT a static popularity
# list. Every laptop in the (budget-filtered) candidate pool gets scored
# on three criteria, then ranked by a weighted blend:
#
#   1. performance_fit  - how well the hardware serves THIS use case
#                          (use-case-specific weighting of CPU/RAM/
#                          storage/GPU/display)
#   2. value_score       - performance delivered per rupee spent,
#                          relative to the rest of the pool (this is
#                          "value for money", not "cheapest")
#   3. longevity_score   - rating, warranty, and CPU generation recency,
#                          as a proxy for whether it'll hold up 3-5 years
#
#   final_score = 0.55 * performance_fit + 0.25 * value_score + 0.20 * longevity_score
#
# All sub-scores are min-max normalized WITHIN the current candidate
# pool (i.e. relative to the other laptops that also fit the budget),
# so the engine adapts to whatever price range the user picks.

CPU_TIER_RANK = {
    "celeron": 0.5, "pentium": 1, "athlon": 1,
    "core i3": 2, "ryzen 3": 2,
    "core i5": 3, "ryzen 5": 3, "m1": 3.5, "snapdragon": 3,
    "core i7": 4, "ryzen 7": 4, "m2": 4.2, "ryzen ai 7": 4.2,
    "core i9": 5, "ryzen 9": 5, "m3": 4.6, "m4": 5, "ryzen ai 9": 5,
}

USE_CASES = {
    "office": {
        "label": "Office & Productivity",
        "description": "Docs, spreadsheets, video calls, browser-heavy work",
        "weights": {"cpu": 0.15, "ram": 0.20, "storage": 0.15, "gpu": 0.00, "display": 0.15, "longevity": 0.35},
        "primary": ["longevity", "ram"],
    },
    "gaming": {
        "label": "Gaming",
        "description": "Modern titles at good settings, sustained load",
        "weights": {"cpu": 0.20, "ram": 0.15, "storage": 0.15, "gpu": 0.40, "display": 0.10, "longevity": 0.00},
        "primary": ["gpu", "cpu"],
    },
    "research": {
        "label": "Research & Data Analysis",
        "description": "Large datasets, statistical tools, many browser tabs / notebooks at once",
        "weights": {"cpu": 0.28, "ram": 0.37, "storage": 0.15, "gpu": 0.05, "display": 0.10, "longevity": 0.05},
        "primary": ["ram", "cpu"],
    },
    "coding": {
        "label": "Programming & Development",
        "description": "IDEs, local builds/compiling, containers, multitasking",
        "weights": {"cpu": 0.30, "ram": 0.30, "storage": 0.20, "gpu": 0.05, "display": 0.05, "longevity": 0.10},
        "primary": ["cpu", "ram"],
    },
    "creative": {
        "label": "Video / Photo Editing & Design",
        "description": "Rendering, exports, large media files, GPU-accelerated tools",
        "weights": {"cpu": 0.25, "ram": 0.20, "storage": 0.20, "gpu": 0.20, "display": 0.15, "longevity": 0.00},
        "primary": ["gpu", "display"],
    },
    "student": {
        "label": "Student",
        "description": "Notes, assignments, browsing, occasional lightweight coding",
        "weights": {"cpu": 0.15, "ram": 0.20, "storage": 0.15, "gpu": 0.00, "display": 0.15, "longevity": 0.35},
        "primary": ["longevity", "ram"],
    },
    "casual": {
        "label": "Casual / Everyday Use",
        "description": "Web browsing, streaming, social media, light multitasking",
        "weights": {"cpu": 0.10, "ram": 0.15, "storage": 0.15, "gpu": 0.00, "display": 0.20, "longevity": 0.40},
        "primary": ["longevity", "display"],
    },
}

DIM_LABELS = {
    "cpu": "processor power", "ram": "RAM", "storage": "storage",
    "gpu": "graphics performance", "display": "display quality", "longevity": "build quality / longevity",
}

# Some frontends label this use case "creator" instead of "creative" -- accept both.
USE_CASE_ALIASES = {"creator": "creative"}


def normalize_use_case(use_case: str) -> str:
    return USE_CASE_ALIASES.get(use_case, use_case)


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _cpu_tier_rank(tier: Optional[str]) -> float:
    if not tier or pd.isna(tier):
        return 1.0
    t = str(tier).lower()
    for key, rank in CPU_TIER_RANK.items():
        if key in t:
            return rank
    return 1.5


USE_CASE_TARGETS = {
    # Sufficiency targets per use case: hitting the target = full marks on
    # that dimension. Going further doesn't add more performance_fit score
    # (there's no benefit to 64GB RAM for office work) -- it just makes the
    # laptop pricier, which is exactly what should hurt its value_score.
    "office":   {"cpu": 11, "ram": 16, "storage": 512,  "gpu": 0,  "display": 1.4},
    "gaming":   {"cpu": 15, "ram": 16, "storage": 1024, "gpu": 11, "display": 2.1},
    "research": {"cpu": 15, "ram": 32, "storage": 512,  "gpu": 3,  "display": 2.1},
    "coding":   {"cpu": 15, "ram": 16, "storage": 512,  "gpu": 3,  "display": 1.4},
    "creative": {"cpu": 15, "ram": 32, "storage": 1024, "gpu": 11, "display": 3.7},
    "student":  {"cpu": 8,  "ram": 8,  "storage": 256,  "gpu": 0,  "display": 1.4},
    "casual":   {"cpu": 6,  "ram": 8,  "storage": 256,  "gpu": 0,  "display": 1.4},
}


def _sufficiency(raw: pd.Series, target: float) -> pd.Series:
    """1.0 once `raw` reaches `target`; scales linearly below it. A target
    of 0 means this dimension isn't required for the use case, so it's
    automatically full marks (its weight will also be 0, so it's inert)."""
    if target <= 0:
        return pd.Series(1.0, index=raw.index)
    return (raw / target).clip(upper=1.0)


def score_pool(df: pd.DataFrame, use_case: str = "office") -> pd.DataFrame:
    """Attach per-dimension 0-1 scores to every row, scored against
    use-case sufficiency targets (not against the single best laptop in
    the pool -- see USE_CASE_TARGETS above for why that distinction matters)."""
    d = df.copy()
    targets = USE_CASE_TARGETS.get(use_case, USE_CASE_TARGETS["office"])

    cpu_tier_score = d["cpu_tier"].apply(_cpu_tier_rank)
    cpu_raw = d["cpu_cores"].fillna(2) * 0.6 + cpu_tier_score * 2 + d["cpu_generation"].fillna(8) * 0.15
    d["dim_cpu"] = _sufficiency(cpu_raw, targets["cpu"])

    d["dim_ram"] = _sufficiency(d["ram_gb"].fillna(0), targets["ram"])

    storage_raw = d["ssd_gb"].fillna(0) + d["hdd_gb"].fillna(0) * 0.3
    d["dim_storage"] = _sufficiency(storage_raw, targets["storage"])

    gpu_raw = d["gpu_vram_gb"].fillna(0) + d["gpu_dedicated"].astype(int) * 3
    d["dim_gpu"] = _sufficiency(gpu_raw, targets["gpu"])

    display_raw = d["resolution_px"].fillna(d["resolution_px"].median()) / 1_000_000 + d["touchscreen"].astype(int) * 0.3
    d["dim_display"] = _sufficiency(display_raw, targets["display"])

    # Longevity has no natural "enough" ceiling (newer/better-rated is
    # always preferable), so it stays relative to the current pool.
    longevity_raw = (
        _normalize(d["rating"].fillna(d["rating"].median())) * 0.5
        + _normalize(d["warranty_years"].fillna(1)) * 0.2
        + _normalize(d["cpu_generation"].fillna(d["cpu_generation"].median())) * 0.3
    )
    d["dim_longevity"] = longevity_raw
    return d


def score_and_rank(df: pd.DataFrame, use_case: str) -> pd.DataFrame:
    profile = USE_CASES.get(use_case, USE_CASES["office"])
    weights = profile["weights"]
    d = score_pool(df, use_case)

    d["performance_fit"] = (
        weights["cpu"] * d["dim_cpu"] + weights["ram"] * d["dim_ram"] + weights["storage"] * d["dim_storage"]
        + weights["gpu"] * d["dim_gpu"] + weights["display"] * d["dim_display"] + weights["longevity"] * d["dim_longevity"]
    )

    value_raw = d["performance_fit"] / (d["price_inr"] / 1000.0)
    d["value_score"] = _normalize(value_raw)
    d["longevity_score"] = d["dim_longevity"]

    d["final_score"] = 0.55 * d["performance_fit"] + 0.25 * d["value_score"] + 0.20 * d["longevity_score"]
    return d.sort_values("final_score", ascending=False)


def _fmt_money(n) -> str:
    return f"\u20b9{int(n):,}"


def build_reasoning(row: pd.Series, pool: pd.DataFrame, use_case: str) -> dict:
    """Grounded, data-driven explanation - every sentence traces back to an
    actual computed score or spec, nothing invented."""
    profile = USE_CASES.get(use_case, USE_CASES["office"])
    weights = profile["weights"]
    primary_dims = profile["primary"]

    # --- 1. Primary use-case fit ---
    rating_txt = f"{row.rating:.0f}/100 user rating" if pd.notna(row.rating) else "no user rating yet"
    fit_bits = []
    if "cpu" in primary_dims:
        fit_bits.append(f"a {row.cpu_tier or row.cpu_brand} processor ({int(row.cpu_cores) if pd.notna(row.cpu_cores) else '?'} cores)")
    if "ram" in primary_dims:
        fit_bits.append(f"{int(row.ram_gb)}GB RAM")
    if "gpu" in primary_dims:
        fit_bits.append(f"{'a dedicated ' + row.gpu_model + (f' ({int(row.gpu_vram_gb)}GB VRAM)' if row.gpu_vram_gb else '') if row.gpu_dedicated else 'integrated graphics (' + str(row.gpu_model) + ')'}")
    if "display" in primary_dims:
        fit_bits.append(f"a {row.display_size_in}\" {row.resolution_label or ''} display" + (" with touch" if row.touchscreen else ""))
    if "longevity" in primary_dims:
        fit_bits.append(f"{rating_txt} and {int(row.warranty_years)}-year warranty")
    use_case_sentence = (
        f"For {profile['label'].lower()}, the top priorities are {' and '.join(DIM_LABELS[d] for d in primary_dims)}. "
        f"This laptop brings {' and '.join(fit_bits)}, which is what actually moves the needle for this workload "
        f"(scored {row.performance_fit:.0%} fit against the rest of the pool)."
    )

    # --- 2. Value-to-performance ---
    pool_avg_price = pool["price_inr"].mean()
    percentile = (pool["performance_fit"] < row.performance_fit).mean() * 100
    if row.price_inr <= pool_avg_price:
        value_sentence = (
            f"At {_fmt_money(row.price_inr)}, it's priced below the average of {_fmt_money(pool_avg_price)} among the "
            f"laptops considered, yet its performance fit beats {percentile:.0f}% of them — strong value, not just a low price tag."
        )
    else:
        value_sentence = (
            f"At {_fmt_money(row.price_inr)} it's above the average of {_fmt_money(pool_avg_price)} among the laptops "
            f"considered, but it outperforms {percentile:.0f}% of the alternatives on the specs that matter for "
            f"{profile['label'].lower()} — the extra spend buys real capability, not just brand markup."
        )

    # --- 3. Longevity / build quality ---
    gen_txt = f"{int(row.cpu_generation)}th-gen" if pd.notna(row.cpu_generation) else "unspecified-gen"
    longevity_sentence = (
        f"Longevity check: {rating_txt}, {int(row.warranty_years)}-year warranty, and a "
        f"{gen_txt} processor — {'recent enough to stay comfortably usable for 3-5 years' if row.dim_longevity >= 0.5 else 'on the older side, so expect it to feel dated sooner than newer picks in this list'}."
    )

    # --- 4. Trade-off vs. a cheaper alternative ---
    cheaper = pool[pool.price_inr < row.price_inr].sort_values("price_inr", ascending=False)
    if len(cheaper) == 0:
        tradeoff_sentence = (
            f"This is already one of the more affordable options that still clears a solid performance bar for "
            f"{profile['label'].lower()} — nothing cheaper in this list matches its {DIM_LABELS[primary_dims[0]]}."
        )
    else:
        alt = cheaper.iloc[0]
        dim_cols = {k: f"dim_{k}" for k in weights}
        gaps = {k: row[dim_cols[k]] - alt[dim_cols[k]] for k in weights if weights[k] > 0}
        worst_dim = max(gaps, key=gaps.get)
        if gaps[worst_dim] > 0.1:
            spec_detail = {
                "cpu": f"{int(alt.cpu_cores) if pd.notna(alt.cpu_cores) else '?'}-core {alt.cpu_tier or alt.cpu_brand} vs {int(row.cpu_cores) if pd.notna(row.cpu_cores) else '?'}-core {row.cpu_tier or row.cpu_brand}",
                "ram": f"{int(alt.ram_gb)}GB vs {int(row.ram_gb)}GB",
                "storage": f"{int(alt.ssd_gb)}GB SSD vs {int(row.ssd_gb)}GB SSD",
                "gpu": f"{alt.gpu_model} vs {row.gpu_model}",
                "display": f"{alt.resolution_label or 'lower-res'} vs {row.resolution_label or 'higher-res'} display",
                "longevity": f"{'%.0f' % alt.rating if pd.notna(alt.rating) else 'no'}/100 rating vs {'%.0f' % row.rating if pd.notna(row.rating) else 'no'}/100",
            }[worst_dim]
            alt_name = alt.model_name.split("(")[0].strip()
            if alt_name.lower().startswith(str(alt.brand).lower()):
                alt_name = alt_name[len(str(alt.brand)):].strip()
            tradeoff_sentence = (
                f"We passed over the cheaper {alt.brand} {alt_name} "
                f"({_fmt_money(alt.price_inr)}) because its {DIM_LABELS[worst_dim]} falls short for "
                f"{profile['label'].lower()}: {spec_detail}. That gap matters more here than the "
                f"{_fmt_money(row.price_inr - alt.price_inr)} price difference."
            )
        else:
            tradeoff_sentence = (
                f"A cheaper option ({alt.brand}, {_fmt_money(alt.price_inr)}) exists in this list, but the specs are "
                f"close enough that this pick's overall score still edges it out on value and longevity combined."
            )

    return {
        "use_case_fit": use_case_sentence,
        "value_to_performance": value_sentence,
        "longevity_build_quality": longevity_sentence,
        "tradeoff_vs_cheaper_alternative": tradeoff_sentence,
    }


def apply_filters(df: pd.DataFrame, filters: "FilterParams") -> pd.DataFrame:
    out = df
    if filters is None:
        return out
    if filters.min_price is not None:
        out = out[out.price_inr >= filters.min_price]
    if filters.max_price is not None:
        out = out[out.price_inr <= filters.max_price]
    if filters.min_ram_gb is not None:
        out = out[out.ram_gb >= filters.min_ram_gb]
    if filters.min_ssd_gb is not None:
        out = out[out.ssd_gb >= filters.min_ssd_gb]
    if filters.brand:
        out = out[out.brand.str.lower() == filters.brand.lower()]
    if filters.cpu_brand:
        out = out[out.cpu_brand.str.lower() == filters.cpu_brand.lower()]
    if filters.gpu_dedicated_only:
        out = out[out.gpu_dedicated == True]  # noqa: E712
    if filters.touchscreen_only:
        out = out[out.touchscreen == True]  # noqa: E712
    if filters.min_rating is not None:
        out = out[out.rating >= filters.min_rating]
    if filters.search:
        q = filters.search.lower()
        out = out[out.model_name.str.lower().str.contains(q) | out.brand.str.lower().str.contains(q)]
    return out


def recommend(filters: "FilterParams", use_case: str, top_n: int = 5) -> List[dict]:
    candidates = apply_filters(LAPTOPS_DF, filters).reset_index(drop=True)
    if len(candidates) == 0:
        return []
    if len(candidates) == 1:
        row = score_pool(candidates, use_case).iloc[0]
        return [{"laptop": row.to_dict(), "final_score": 1.0, "score_breakdown": {}, "reasoning": {
            "use_case_fit": "Only one laptop matches these filters, shown without comparative scoring.",
            "value_to_performance": "", "longevity_build_quality": "", "tradeoff_vs_cheaper_alternative": "",
        }} for _ in [0]]

    ranked = score_and_rank(candidates, use_case)
    top = ranked.head(top_n)
    results = []
    for _, row in top.iterrows():
        reasoning = build_reasoning(row, ranked, use_case)
        results.append({
            "laptop": row.drop(labels=[c for c in row.index if c.startswith("dim_")]).to_dict(),
            "final_score": round(float(row.final_score), 4),
            "score_breakdown": {
                "performance_fit": round(float(row.performance_fit), 4),
                "value_score": round(float(row.value_score), 4),
                "longevity_score": round(float(row.longevity_score), 4),
            },
            "reasoning": reasoning,
        })
    return results


# ============================================================
# 3. API SCHEMAS
# ============================================================

class FilterParams(BaseModel):
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_ram_gb: Optional[int] = None
    min_ssd_gb: Optional[int] = None
    brand: Optional[str] = None
    cpu_brand: Optional[str] = None
    gpu_dedicated_only: Optional[bool] = None
    touchscreen_only: Optional[bool] = None
    min_rating: Optional[float] = None
    search: Optional[str] = None


class RecommendRequest(BaseModel):
    filters: Optional[FilterParams] = None
    use_case: str = Field("office", description="office | gaming | research | coding | creative")
    top_n: int = 5


class CompareRequest(BaseModel):
    laptop_ids: List[int] = Field(..., min_length=1, max_length=3)
    use_case: Optional[str] = Field(None, description="If given, include scoring/reasoning for this use case")


# ============================================================
# 4. FASTAPI APP
# ============================================================

app = FastAPI(title="Laptop Recommendation API (Hackathon Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_for_json(record: dict) -> dict:
    """Replace NaN/NaT with None so FastAPI's JSON encoder doesn't choke."""
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


@app.get("/")
def root():
    return {"message": "Laptop Recommendation API is running. See /docs to try it out.", "total_laptops": len(LAPTOPS_DF)}


@app.get("/use-cases")
def get_use_cases():
    return {k: {"label": v["label"], "description": v["description"]} for k, v in USE_CASES.items()}


@app.get("/filter-options")
def get_filter_options():
    return {
        "brands": sorted(LAPTOPS_DF.brand.dropna().unique().tolist()),
        "cpu_brands": sorted(LAPTOPS_DF.cpu_brand.dropna().unique().tolist()),
        "price_min": int(LAPTOPS_DF.price_inr.min()),
        "price_max": int(LAPTOPS_DF.price_inr.max()),
    }


@app.get("/laptops")
def list_laptops(
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    min_ram_gb: Optional[int] = None,
    min_ssd_gb: Optional[int] = None,
    brand: Optional[str] = None,
    cpu_brand: Optional[str] = None,
    gpu_dedicated_only: Optional[bool] = None,
    touchscreen_only: Optional[bool] = None,
    min_rating: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    filters = FilterParams(
        min_price=min_price, max_price=max_price, min_ram_gb=min_ram_gb, min_ssd_gb=min_ssd_gb,
        brand=brand, cpu_brand=cpu_brand, gpu_dedicated_only=gpu_dedicated_only,
        touchscreen_only=touchscreen_only, min_rating=min_rating, search=search,
    )
    result = apply_filters(LAPTOPS_DF, filters).iloc[offset: offset + limit]
    return [_clean_for_json(r) for r in result.to_dict(orient="records")]


@app.get("/laptops/{laptop_id}")
def get_laptop(laptop_id: int):
    match = LAPTOPS_DF[LAPTOPS_DF.laptop_id == laptop_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Laptop not found")
    return _clean_for_json(match.iloc[0].to_dict())


@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    use_case = normalize_use_case(req.use_case)
    if use_case not in USE_CASES:
        raise HTTPException(status_code=400, detail=f"use_case must be one of {list(USE_CASES)} (or alias: {list(USE_CASE_ALIASES)})")
    results = recommend(filters=req.filters, use_case=use_case, top_n=req.top_n)
    for r in results:
        r["laptop"] = _clean_for_json(r["laptop"])
    return results


@app.post("/compare")
def compare_laptops(req: CompareRequest):
    rows = LAPTOPS_DF[LAPTOPS_DF.laptop_id.isin(req.laptop_ids)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="None of the given laptop_ids were found")

    if req.use_case:
        use_case = normalize_use_case(req.use_case)
        if use_case not in USE_CASES:
            raise HTTPException(status_code=400, detail=f"use_case must be one of {list(USE_CASES)} (or alias: {list(USE_CASE_ALIASES)})")
        scored = score_and_rank(rows, use_case)
        out = []
        for _, row in scored.iterrows():
            out.append({
                "laptop": _clean_for_json(row.drop(labels=[c for c in row.index if c.startswith("dim_")]).to_dict()),
                "final_score": round(float(row.final_score), 4),
                "score_breakdown": {
                    "performance_fit": round(float(row.performance_fit), 4),
                    "value_score": round(float(row.value_score), 4),
                    "longevity_score": round(float(row.longevity_score), 4),
                },
                "reasoning": build_reasoning(row, scored, use_case),
            })
        out.sort(key=lambda r: -r["final_score"])
        return out
    else:
        return [{"laptop": _clean_for_json(r)} for r in rows.to_dict(orient="records")]


# ============================================================
# 5. RUN (this is what makes F5 in IDLE work)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\nLoaded {len(LAPTOPS_DF)} laptops. Starting server...")
    print("Open http://127.0.0.1:8000/docs in your browser once it starts.\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
