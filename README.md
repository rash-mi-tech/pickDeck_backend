# Multi-Category Recommendation Backend

Extends the original laptop-only engine to 5 categories: **laptops, phones,
headphones/earbuds, smartwatches/fitness trackers, and tablets** — same
scoring philosophy (performance fit + value-to-performance + longevity,
never sorted by price alone) across all of them.

## Files
- `main.py` — FastAPI app, category-aware endpoints
- `engine.py` — the shared scoring math (works for any category via config)
- `category_configs.py` — per-category dimensions, use cases, sufficiency targets
- `data/*_clean.csv` — cleaned datasets (already processed, ready to use)
- `data/laptop.csv`, `Mobiles_Dataset__2025_.csv`, etc. — original raw files, kept for reference

## Run it
```
pip install -r requirements.txt
python main.py
```
Then open http://127.0.0.1:8000/docs to explore every endpoint interactively.

## API shape (changed from the laptop-only version)

Every endpoint now takes a `category` parameter:

| Endpoint | Notes |
|---|---|
| `GET /categories` | List all 5 categories + item counts |
| `GET /use-cases?category=phone` | Use cases available for that category |
| `GET /filter-options?category=phone` | Brand list, price range for that category |
| `GET /items?category=phone&...` | List/filter/search |
| `GET /items/{category}/{item_id}` | Single item detail |
| `POST /recommend` | Body now includes `"category": "phone"` alongside `filters`, `use_case`, `top_n` |
| `POST /compare` | Body now includes `"category"` and `item_ids` (was `laptop_ids`) |

## Use cases per category

- **Laptop**: office, gaming, coding, creative, student
- **Phone**: gaming, camera, battery_life, budget, business
- **Headphone**: gym_workout, travel, studio_critical, budget
- **Watch**: fitness_tracking, fashion_smartwatch, budget
- **Tablet**: student, creative, entertainment, business

## Known data gaps (be upfront about these if asked)

- **Phones have no storage/ROM column** in the source dataset — only RAM.
  Storage isn't a scoring factor for phones as a result.
- **Phones have no rating data at all.** Longevity for phones is based on
  launch year (recency) + RAM headroom instead of user rating.
- **Headphones have no battery life, ANC, or warranty data.** Scoring leans
  on rating + review count (as a "proven in the market" signal) and how well
  the physical type (earbuds/on-ear/etc.) matches the use case.
- **Watches**: 56 of 610 rows are missing a rating — the code fills these
  with the category median rather than dropping them.
- **Headphones**: ~14% of raw rows had a shifted/missing Color field (same
  class of bug the laptop dataset had). Recovered via model-name pattern
  matching where possible; genuinely unrecoverable rows were dropped
  (33 out of 1000).

## What's NOT built yet

- The frontend category-selector page (pick a category → get a page like
  the laptop finder, scoped to that category)
- The Smart Ecosystem / Bundle Builder feature
- Speakers and power banks (no usable dataset found yet)

## Row counts after cleaning

| Category | Raw rows | Usable |
|---|---|---|
| Laptop | 920 | 910 |
| Phone | 930 | 930 |
| Headphone | 1000 | 967 |
| Watch | 610 | 610 |
| Tablet | 390 | 365 |
