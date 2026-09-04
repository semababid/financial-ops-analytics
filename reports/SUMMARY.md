# Financial Operations Analytics: Findings Summary

Analysis of the Olist Brazilian e-commerce marketplace (Jan 2017 to Aug 2018,
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

The winning model projects a **flat ~R$855K/month** over the next 6 months, which is
the honest read on a plateaued series with only one year of seasonal history.
→ `figures/07`, `08`

## 3. Churn & retention

**RFM segmentation** of 94.7k customers. Frequency is degenerate here, since ~97% of
customers order exactly once, so a classic 5x5 RFM would fabricate loyalty that
isn't there. Instead loyalty is a binary repeat flag (its own segment), and the
one-time majority is segmented on recency x monetary:

| Segment | Customers | Revenue share | Who they are |
|---------|-----------|---------------|--------------|
| Champions       | 14,709 | 29.4% | recent, high-spend (prime to convert to repeat) |
| At Risk         | 13,814 | 28.6% | were high-spend, now gone quiet |
| Needs Attention | 18,329 | 17.8% | middle recency band |
| Hibernating     | 22,879 |  9.5% | old and low-value |
| Promising       | 22,087 |  9.2% | recent but low-spend |
| Loyal / Repeat  |  2,876 |  5.6% | the ~3% who actually reorder |

**Repeat-purchase model** (predict a 2nd order from 1st-order features):
ROC-AUC 0.61, a weak but real signal. Framed as targeting rather than
classification, the **top propensity decile finds repeaters 1.8x better than
random**, and the top 3 deciles capture ~42% of all repeaters.
Strongest drivers: order value, freight, **delivery delay**, and review score.
→ `figures/09`, `10`, `11`

**Takeaway:** delivery experience and review score predict repeat purchase.
Operational quality is a retention driver, not just a cost.

## 4. Profitability (marketplace contribution model)

Modeled as commission (15% of price) minus payment processing (2.5% of paid).
Rates are assumptions; the *relative* rankings are the point.

| Metric | Value |
|--------|-------|
| GMV | R$ 13.4M |
| Commission revenue | R$ 2.0M |
| Contribution | **R$ 1.6M (12.1% of GMV)** |
| Freight handled | **R$ 2.2M (16.6% of GMV)** |

- **Freight (16.6%) exceeds contribution (12.1%).** Logistics is the dominant
  margin lever, not category mix.
- Margin % is ~flat across categories (uniform commission); the real spread is
  **freight drag**: electronics (63% of price), telephony (43%), food/drink
  (41%), all cheap and heavy items.
- **SP** has the lowest freight burden (13.8%) thanks to its logistics-hub
  location; remote states run 17-18%.
- Seller concentration: the **top 10% of sellers drive 46%** of contribution.
→ `figures/12`, `13`

## 5. Shipping distance: why freight costs what it does

Section 4 established that freight is the biggest drag on margin, but not why.
The geolocation table maps zip prefixes to coordinates, so I attached seller and
customer coordinates to every order item and computed the haversine distance
between them. That matched 99.5% of items. The median shipment travels 432 km,
which is the shape of the problem: this marketplace ships long routes across a
very large country.

Freight, delivery time and lateness all rise with distance:

| Distance | Avg freight | Freight/price | Avg delivery | Late rate |
|----------|------------|---------------|--------------|-----------|
| <50 km     | R$ 11.47 | 17.7% |  5.7 d |  4.4% |
| 50-150     | R$ 13.21 | 18.1% |  7.2 d |  4.6% |
| 150-400    | R$ 18.39 | 22.0% | 11.2 d |  6.2% |
| 400-800    | R$ 20.48 | 24.7% | 13.0 d |  6.9% |
| 800-1500   | R$ 23.66 | 26.5% | 15.3 d |  7.3% |
| 1500+      | R$ 35.74 | 33.8% | 20.1 d | 11.3% |

Across that range freight roughly triples (R$11 to R$36), delivery time goes
from 5.7 to 20.1 days, and the late rate goes from 4.4% to 11.3%.

An OLS that controls for product weight puts a number on it:

```
freight ≈ +R$ 1.06 per 100 km  (t=193)
          +R$ 2.59 per kg      (t=301)
          R² = 0.533
```

Distance and weight together account for 53% of the variance in freight. At the
state level, average distance and freight burden correlate at 0.87, and that is
what explains SP's advantage: its average shipment travels 247 km against
1,345 km for Bahia. So SP's freight burden sits at 13.8% while the northern
states (MA, RO, PI) run between 22% and 26%.
→ `figures/14`, `15`

**Takeaway:** the freight problem is really a distance problem, and it costs
twice. Long routes are expensive and they also arrive late, and late delivery is
one of the stronger repeat-purchase signals from section 3. Regional fulfilment
would help margin and retention at once.

---

## Recommendations

1. **Retention program** targeting the model's top propensity deciles plus the
   "Champions" and "At Risk" segments (recent high-spenders to convert, and
   lapsing high-value customers to win back), together ~58% of revenue.
2. **Regional fulfilment in the Northeast.** Freight scales at ~R$1.06/100 km,
   so the states shipping 1,300 to 2,300 km (BA, PE, CE, MA) are where a second
   distribution hub pays for itself: it would cut both the 20-26% freight burden
   and the 15 to 20 day delivery times. Also reconsider free shipping on low-value
   heavy categories (electronics, telephony, food/drink).
3. **Delivery SLA as a retention KPI.** Late delivery is a top churn driver, and
   on-time delivery compounds into repeat revenue.

## Caveats

- 20-month dense window; forecasts beyond ~6 months are low-confidence.
- Commission/payment rates are illustrative assumptions (see `src/config.py`).
- Repeat-purchase signal is weak (AUC 0.61): use for prioritization, not gating.
