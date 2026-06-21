# Financial Operations Analytics — Findings Summary

Analysis of the Olist Brazilian e-commerce marketplace (Jan 2017 – Aug 2018,
the dense data window). All figures are reproducible via
`python scripts/run_pipeline.py`.

---

## 1. Business overview (EDA)

| Metric | Value |
|--------|-------|
| Realized revenue (GMV) | **R$ 13.4M** |
| Realized orders | 97,896 |
| Unique customers | 94,694 |
| Repeat-buyer share | **3.0%** |
| Avg order value | R$ 137 |
| Median delivery time | 10 days |
| Late-delivery share | 6.6% |
| Top category | `health_beauty` (9.3% of revenue) |

Revenue grew ~6x from Jan 2017 to a ~R$1M/month plateau in 2018, with a clear
Nov 2017 Black-Friday spike. → `figures/01`, `02`, `03`, `04`, `05`, `06`

**Takeaway:** acquisition-led growth that has plateaued; almost no repeat
purchasing. Retention is the biggest untapped lever.

## 2. Revenue forecasting

A model bake-off (seasonal-naive baseline vs. damped Holt-Winters vs. 4 SARIMA
orders) was backtested on a 4-month holdout.

| Model | Holdout MAPE |
|-------|--------------|
| **SARIMA(0,1,1)** | **10.7%** ← selected |
| SARIMA(1,1,0) | 11.2% |
| Holt-Winters (damped) | 28.3% |
| Baseline | 77.1% |

The winning model projects a **flat ~R$855K/month** over the next 6 months — the
honest read on a plateaued series with only one year of seasonal history.
→ `figures/07`, `08`

## 3. Churn & retention

**RFM segmentation** of 94.7k customers:

| Segment | Customers | Revenue share |
|---------|-----------|---------------|
| At Risk | 22,540 | 24.1% |
| Loyal | 19,069 | 19.6% |
| Champions | 15,207 | 17.1% |
| Hibernating | 15,172 | 15.9% |
| New / Promising | 15,229 | 15.8% |
| Needs Attention | 7,477 | 7.4% |

**Repeat-purchase model** (predict a 2nd order from 1st-order features):
ROC-AUC 0.61 — weak but real signal. Framed as targeting rather than
classification, the **top propensity decile finds repeaters 1.8x better than
random**, and the top 3 deciles capture 45% of all repeaters.
Strongest drivers: order value, freight, **delivery delay**, and review score.
→ `figures/09`, `10`, `11`

**Takeaway:** delivery experience and review score predict repeat purchase —
operational quality is a retention driver, not just a cost.

## 4. Profitability (marketplace contribution model)

Modeled as commission (15% of price) minus payment processing (2.5% of paid).
Rates are assumptions; the *relative* rankings are the point.

| Metric | Value |
|--------|-------|
| GMV | R$ 13.4M |
| Commission revenue | R$ 2.0M |
| Contribution | **R$ 1.6M (12.1% of GMV)** |
| Freight handled | **R$ 2.2M (16.6% of GMV)** |

- **Freight (16.6%) exceeds contribution (12.1%)** — logistics is the dominant
  margin lever, not category mix.
- Margin % is ~flat across categories (uniform commission); the real spread is
  **freight drag**: electronics (63% of price), telephony (43%), food/drink
  (41%) — cheap, heavy items.
- **SP** has the lowest freight burden (13.8%) thanks to its logistics-hub
  location; remote states run 17–18%.
- Seller concentration: the **top 10% of sellers drive 46%** of contribution.
→ `figures/12`, `13`

---

## Recommendations

1. **Retention program** targeting the model's top deciles + "At Risk" /
   "Loyal" RFM segments, which together hold the majority of revenue.
2. **Logistics investment** (freight is the #1 margin drag) — regional fulfilment
   to cut the 17–18% freight burden in non-SP states; reconsider free-shipping
   on low-value heavy categories (electronics, telephony, food/drink).
3. **Delivery SLA as a retention KPI** — late delivery is a top churn driver;
   on-time delivery compounds into repeat revenue.

## Caveats

- 20-month dense window; forecasts beyond ~6 months are low-confidence.
- Commission/payment rates are illustrative assumptions (see `src/config.py`).
- Repeat-purchase signal is weak (AUC 0.61): use for prioritization, not gating.
