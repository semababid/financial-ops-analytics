# Financial Operations Analytics

Revenue forecasting, churn, and profitability analysis on the
**Brazilian E-Commerce (Olist)** public dataset.

> Olist is a Brazilian marketplace connecting small sellers to large
> storefronts. The dataset covers ~100k orders from 2016–2018, with order
> items, payments, reviews, customers, sellers, products and geolocation.

## Goals

1. **Revenue forecasting** — model monthly GMV / revenue and project forward.
2. **Churn analysis** — segment customers (RFM) and predict repeat purchase.
3. **Profitability** — margins by category, seller and region; freight and
   payment economics.

## Project layout

```
financial-ops-analytics/
├── data/
│   ├── raw/          # 9 Olist CSVs (git-ignored, ~120 MB)
│   └── processed/    # cleaned parquet tables (git-ignored)
├── src/
│   ├── config.py         # paths + business constants
│   ├── data_loader.py    # typed loaders for each raw CSV
│   ├── data_cleaning.py  # builds master / order-level / monthly tables
│   ├── revenue_forecast.py
│   ├── churn_analysis.py
│   └── profitability.py
├── scripts/          # runnable entry points
├── reports/figures/  # generated charts
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# Place the Kaggle "Brazilian E-Commerce by Olist" archive.zip, then:
unzip archive.zip -d data/raw

# Build processed tables (parquet) from raw CSVs:
python -m src.data_cleaning
```

## Data model (processed)

| Table | Grain | Use |
|-------|-------|-----|
| `orders_master`   | one row per **order item** | category / seller / profitability |
| `order_level`     | one row per **order**      | churn, forecasting inputs |
| `monthly_revenue` | one row per **month**      | time-series forecasting |

**Modeling conventions**
- Revenue is recognized from item `price` (excludes freight, installments, vouchers).
- `customer_unique_id` is the true customer; `customer_id` is per-order.
- Only `delivered` / `shipped` / `invoiced` orders count as realized revenue.

## Status

- [x] Phase 0 — scaffold
- [x] Phase 1 — data ingestion & cleaning pipeline
- [x] Phase 2 — exploratory data analysis (`python -m src.eda`)
- [ ] Phase 3 — revenue forecasting
- [ ] Phase 4 — churn analysis
- [ ] Phase 5 — profitability analysis

## Dataset

Kaggle: *Brazilian E-Commerce Public Dataset by Olist* (CC BY-NC-SA 4.0).
Raw CSVs are not committed; download them from Kaggle and unzip into `data/raw/`.
