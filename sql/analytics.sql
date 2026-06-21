-- Financial Operations Analytics — core metrics in SQL (DuckDB).
-- These queries run directly against the processed parquet tables and
-- reproduce the headline numbers from the Python analysis, demonstrating the
-- same logic in SQL. Run with: python -m src.sql_runner
--
-- Placeholders {order_level} and {orders_master} are filled by the runner with
-- the parquet paths (DuckDB reads parquet as a table via read_parquet).

-- name: monthly_revenue
-- Monthly realized revenue, orders and unique customers.
SELECT
    order_month,
    ROUND(SUM(items_value), 2)        AS revenue,
    COUNT(*)                          AS orders,
    COUNT(DISTINCT customer_unique_id) AS customers,
    ROUND(SUM(items_value) / COUNT(*), 2) AS avg_order_value
FROM {order_level}
WHERE is_valid_revenue
GROUP BY order_month
ORDER BY order_month;

-- name: repeat_rate
-- Share of customers who placed more than one valid order.
WITH per_customer AS (
    SELECT customer_unique_id, COUNT(*) AS n_orders
    FROM {order_level}
    WHERE is_valid_revenue
    GROUP BY customer_unique_id
)
SELECT
    COUNT(*)                                          AS customers,
    SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)     AS repeat_customers,
    ROUND(100.0 * AVG(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END), 2) AS repeat_rate_pct
FROM per_customer;

-- name: top_categories
-- Top 10 categories by realized revenue, with revenue share.
SELECT
    category,
    ROUND(SUM(item_revenue), 2) AS revenue,
    COUNT(*)                    AS items,
    ROUND(100.0 * SUM(item_revenue) / SUM(SUM(item_revenue)) OVER (), 2) AS revenue_share_pct
FROM {orders_master}
WHERE is_valid_revenue
GROUP BY category
ORDER BY revenue DESC
LIMIT 10;

-- name: state_profitability
-- Contribution and freight burden by customer state (top 10 by revenue).
-- Mirrors the marketplace model: 15% commission, 2.5% payment fee.
SELECT
    customer_state                                    AS state,
    ROUND(SUM(items_value), 2)                        AS gmv,
    ROUND(SUM(0.15 * items_value
            - 0.025 * (items_value + freight_value)), 2) AS contribution,
    ROUND(100.0 * SUM(freight_value) / SUM(items_value), 1) AS freight_burden_pct
FROM {order_level}
WHERE is_valid_revenue
GROUP BY customer_state
ORDER BY gmv DESC
LIMIT 10;

-- name: delivery_vs_review
-- Does on-time delivery relate to review scores? Bucket by delay.
SELECT
    CASE
        WHEN delivery_delay_days <= -5 THEN '1. >=5d early'
        WHEN delivery_delay_days <   0 THEN '2. early'
        WHEN delivery_delay_days =   0 THEN '3. on time'
        WHEN delivery_delay_days <=  5 THEN '4. <=5d late'
        ELSE                                '5. >5d late'
    END                                  AS delivery_bucket,
    COUNT(*)                             AS orders,
    ROUND(AVG(review_score), 2)          AS avg_review_score
FROM {order_level}
WHERE is_valid_revenue
  AND delivery_delay_days IS NOT NULL
  AND review_score IS NOT NULL
GROUP BY delivery_bucket
ORDER BY delivery_bucket;
