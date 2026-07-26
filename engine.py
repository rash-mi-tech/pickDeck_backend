"""
engine.py
---------
One scoring engine, driven by category_configs.py, used by every category.
Laptops keep their original composite dimension formulas (CPU tier rank,
GPU VRAM + dedicated bonus, etc.) since those are already tested and proven;
every other category uses a simpler direct-column sufficiency formula.

Same MCDA framework everywhere: final_score = 0.55*performance_fit +
0.25*value_score + 0.20*longevity_score. See laptop's main.py for the
original rationale -- it applies identically here.
"""

import numpy as np
import pandas as pd

from category_configs import CATEGORY_CONFIGS

CPU_TIER_RANK = {
    "celeron": 0.5, "pentium": 1, "athlon": 1,
    "core i3": 2, "ryzen 3": 2,
    "core i5": 3, "ryzen 5": 3, "m1": 3.5, "snapdragon": 3,
    "core i7": 4, "ryzen 7": 4, "m2": 4.2, "ryzen ai 7": 4.2,
    "core i9": 5, "ryzen 9": 5, "m3": 4.6, "m4": 5, "ryzen ai 9": 5,
}


def _cpu_tier_rank(tier):
    if not tier or pd.isna(tier):
        return 1.0
    t = str(tier).lower()
    for key, rank in CPU_TIER_RANK.items():
        if key in t:
            return rank
    return 1.5


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _sufficiency(raw: pd.Series, target: float) -> pd.Series:
    if target is None or target <= 0:
        return pd.Series(1.0, index=raw.index)
    return (raw / target).clip(upper=1.0)


def _compute_dims_laptop(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    d = df.copy()
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
    longevity_raw = (
        _normalize(d["rating"].fillna(d["rating"].median())) * 0.5
        + _normalize(d["warranty_years"].fillna(1)) * 0.2
        + _normalize(d["cpu_generation"].fillna(d["cpu_generation"].median())) * 0.3
    )
    d["dim_longevity"] = longevity_raw
    return d


def _compute_dims_generic(df: pd.DataFrame, category: str, targets: dict) -> pd.DataFrame:
    config = CATEGORY_CONFIGS[category]
    d = df.copy()
    for dim_key, dim_meta in config["dims"].items():
        col = dim_meta.get("col", dim_key)
        raw = pd.to_numeric(d[col], errors="coerce").fillna(0)
        d[f"dim_{dim_key}"] = _sufficiency(raw, targets.get(dim_key))

    # Longevity: weighted normalize of whatever trust/futureproofing columns
    # this category actually has data for (see category_configs.py comments
    # on why each category's formula differs).
    formula = config["longevity_formula"]
    longevity_raw = pd.Series(0.0, index=d.index)
    for col, weight in formula.items():
        series = pd.to_numeric(d[col], errors="coerce")
        series = series.fillna(series.median())
        longevity_raw += _normalize(series) * weight
    d["dim_longevity"] = longevity_raw
    return d


def score_pool(df: pd.DataFrame, category: str, use_case: str) -> pd.DataFrame:
    config = CATEGORY_CONFIGS[category]
    uc = config["use_cases"][use_case]
    targets = uc["targets"]

    if category == "laptop":
        d = _compute_dims_laptop(df, targets)
    else:
        d = _compute_dims_generic(df, category, targets)
    return d


def score_and_rank(df: pd.DataFrame, category: str, use_case: str) -> pd.DataFrame:
    config = CATEGORY_CONFIGS[category]
    uc = config["use_cases"][use_case]
    weights = uc["weights"]
    d = score_pool(df, category, use_case)

    dim_keys = list(config["dims"].keys()) if category != "laptop" else ["cpu", "ram", "storage", "gpu", "display"]
    dim_keys = dim_keys + ["longevity"]  # longevity also contributes to performance_fit via its use-case weight
    perf = pd.Series(0.0, index=d.index)
    for dim_key in dim_keys:
        w = weights.get(dim_key, 0)
        if w:
            perf += w * d[f"dim_{dim_key}"]
    d["performance_fit"] = perf

    price_col = config["price_col"]
    value_raw = d["performance_fit"] / (pd.to_numeric(d[price_col], errors="coerce").fillna(1) / 1000.0)
    d["value_score"] = _normalize(value_raw)
    d["longevity_score"] = d["dim_longevity"]

    d["final_score"] = 0.55 * d["performance_fit"] + 0.25 * d["value_score"] + 0.20 * d["longevity_score"]
    return d.sort_values("final_score", ascending=False)


def _fmt_money(n) -> str:
    return f"\u20b9{int(n):,}"


def build_reasoning(row: pd.Series, pool: pd.DataFrame, category: str, use_case: str) -> dict:
    """Grounded, data-driven explanation shared across every category -- the
    same 4-part structure (use-case fit, value, longevity, trade-off) as the
    original laptop engine, generalized to whatever dims this category has."""
    config = CATEGORY_CONFIGS[category]
    uc = config["use_cases"][use_case]
    weights = uc["weights"]
    primary_dims = uc["primary"]
    price_col = config["price_col"]
    dim_labels = {**{k: v["label"] for k, v in config["dims"].items()}, "longevity": "longevity"}

    is_laptop = category == "laptop"
    dim_cols = {k: f"dim_{k}" for k in (list(config["dims"].keys()) + ["longevity"])}

    fit_bits = []
    for dim_key in primary_dims:
        label = dim_labels.get(dim_key, dim_key)
        if dim_key == "longevity":
            continue  # described in its own section below
        col = config["dims"].get(dim_key, {}).get("col", dim_key) if not is_laptop else None
        if col and col in row.index:
            val = row[col]
            fit_bits.append(f"{val:g} in {label}" if isinstance(val, (int, float)) else f"{val} {label}")
        else:
            fit_bits.append(label)
    if not fit_bits:
        fit_bits = [dim_labels.get(d, d) for d in primary_dims]

    use_case_sentence = (
        f"For {uc['label'].lower()}, the top priorities are {' and '.join(dim_labels.get(d, d) for d in primary_dims)}. "
        f"This item scores {row['performance_fit']:.0%} fit against the rest of the pool on exactly those dimensions."
    )

    pool_avg_price = pool[price_col].mean()
    percentile = (pool["performance_fit"] < row["performance_fit"]).mean() * 100
    if row[price_col] <= pool_avg_price:
        value_sentence = (
            f"At {_fmt_money(row[price_col])}, it's priced below the average of {_fmt_money(pool_avg_price)} among "
            f"items considered, yet its performance fit beats {percentile:.0f}% of them."
        )
    else:
        value_sentence = (
            f"At {_fmt_money(row[price_col])} it's above the average of {_fmt_money(pool_avg_price)}, but it "
            f"outperforms {percentile:.0f}% of the alternatives on the specs that matter for {uc['label'].lower()}."
        )

    longevity_sentence = f"Longevity score: {row['longevity_score']:.0%} relative to the rest of the pool, based on this category's available trust signals (rating, and category-specific recency/durability proxies)."

    cheaper = pool[pool[price_col] < row[price_col]].sort_values(price_col, ascending=False)
    if len(cheaper) == 0:
        tradeoff_sentence = f"This is already one of the more affordable options that still clears a solid bar for {uc['label'].lower()}."
    else:
        alt = cheaper.iloc[0]
        gaps = {k: row[dim_cols[k]] - alt[dim_cols[k]] for k in weights if weights.get(k, 0) > 0 and dim_cols[k] in row.index}
        if gaps:
            worst_dim = max(gaps, key=gaps.get)
            if gaps[worst_dim] > 0.1:
                tradeoff_sentence = (
                    f"We passed over the cheaper {alt['brand']} {str(alt['model_name']).split('(')[0].strip()} "
                    f"({_fmt_money(alt[price_col])}) because its {dim_labels.get(worst_dim, worst_dim)} falls short "
                    f"for {uc['label'].lower()}. That gap matters more here than the "
                    f"{_fmt_money(row[price_col] - alt[price_col])} price difference."
                )
            else:
                tradeoff_sentence = (
                    f"A cheaper option ({alt['brand']}, {_fmt_money(alt[price_col])}) exists, but the specs are "
                    f"close enough that this pick's overall score still edges it out on value and longevity combined."
                )
        else:
            tradeoff_sentence = "No comparable cheaper alternative with matching data was found in this pool."

    return {
        "use_case_fit": use_case_sentence,
        "value_to_performance": value_sentence,
        "longevity_build_quality": longevity_sentence,
        "tradeoff_vs_cheaper_alternative": tradeoff_sentence,
    }
