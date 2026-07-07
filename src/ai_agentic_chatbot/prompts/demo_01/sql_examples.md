## EXAMPLE QUERY PAIRS

---

### CUSTOMER QUERIES

---

**Q: Show all active customers**
```sql
SELECT code, customer_name, short_name, credit_status
FROM customer
WHERE activation_status = 'ACTIVE'
ORDER BY customer_name ASC
LIMIT 100;
```

---

**Q: List customers on credit hold**
```sql
SELECT code, customer_name, credit_status, activation_status
FROM customer
WHERE credit_status = 'HOLD'
ORDER BY customer_name ASC;
```

---

**Q: How many active customers do we have?**
```sql
SELECT COUNT(*) AS active_customer_count
FROM customer
WHERE activation_status = 'ACTIVE';
```

---

**Q: Show customers who have never placed an order**
```sql
SELECT c.code, c.customer_name, c.activation_status
FROM customer c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL
ORDER BY c.customer_name ASC;
```

---

**Q: Top 10 customers by revenue this year**
```sql
SELECT customer_name,
       customer_code,
       SUM(quantity * unit_price)      AS revenue,
       COUNT(DISTINCT reference_no)    AS order_count,
       SUM(quantity)                   AS total_units
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name, customer_code
ORDER BY revenue DESC
LIMIT 10;
```

---

**Q: Top 10 customers by revenue last year**
```sql
SELECT customer_name,
       customer_code,
       SUM(quantity * unit_price)   AS revenue,
       COUNT(DISTINCT reference_no) AS order_count
FROM v_sales_summary
WHERE order_date BETWEEN
    date_trunc('year', CURRENT_DATE - INTERVAL '1 year')
    AND date_trunc('year', CURRENT_DATE) - INTERVAL '1 day'
GROUP BY customer_name, customer_code
ORDER BY revenue DESC
LIMIT 10;
```

---

**Q: Which customers have not ordered in the last 90 days?**
```sql
SELECT c.code, c.customer_name, c.activation_status,
       MAX(o.created_at::date) AS last_order_date
FROM customer c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.code, c.customer_name, c.activation_status
HAVING MAX(o.created_at::date) < CURRENT_DATE - INTERVAL '90 days'
    OR MAX(o.created_at::date) IS NULL
ORDER BY last_order_date ASC NULLS FIRST;
```

---

**Q: Show customer revenue comparison this year vs last year**
```sql
SELECT customer_name,
       SUM(CASE WHEN order_date >= date_trunc('year', CURRENT_DATE)
                THEN quantity * unit_price ELSE 0 END) AS revenue_this_year,
       SUM(CASE WHEN order_date BETWEEN
                    date_trunc('year', CURRENT_DATE - INTERVAL '1 year')
                    AND date_trunc('year', CURRENT_DATE) - INTERVAL '1 day'
                THEN quantity * unit_price ELSE 0 END) AS revenue_last_year
FROM v_sales_summary
GROUP BY customer_name
ORDER BY revenue_this_year DESC
LIMIT 50;
```

---

### ORDER QUERIES

---

**Q: Show all pending orders**
```sql
SELECT order_no, customer_name, grand_total,
       delivery_method, delivery_date, created_at::date AS order_date
FROM v_sales_summary
WHERE order_status = 'PENDING'
GROUP BY order_no, customer_name, grand_total,
         delivery_method, delivery_date, created_at::date
ORDER BY created_at DESC
LIMIT 100;
```

---

**Q: How many orders were placed today?**
```sql
SELECT COUNT(DISTINCT order_no) AS order_count
FROM orders
WHERE created_at::date = CURRENT_DATE;
```

---

**Q: Show orders placed this month with their status**
```sql
SELECT order_no, customer_name, order_status,
       grand_total, delivery_method, delivery_date,
       created_at::date AS order_date
FROM v_sales_summary
WHERE created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY order_no, customer_name, order_status,
         grand_total, delivery_method, delivery_date, created_at::date
ORDER BY created_at DESC;
```

---

**Q: How many orders are in each status this month?**
```sql
SELECT order_status,
       COUNT(DISTINCT order_no) AS order_count
FROM orders
WHERE created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY order_status
ORDER BY order_count DESC;
```

---

**Q: Show all void orders this year**
```sql
SELECT order_no, customer_name, grand_total,
       created_at::date AS order_date, final_comment
FROM v_sales_summary
WHERE order_status = 'VOID'
AND created_at >= date_trunc('year', CURRENT_DATE)
GROUP BY order_no, customer_name, grand_total,
         created_at::date, final_comment
ORDER BY created_at DESC;
```

---

**Q: What is the total value of pending orders?**
```sql
SELECT COUNT(DISTINCT order_no)  AS pending_order_count,
       SUM(grand_total)          AS total_pending_value
FROM orders
WHERE order_status = 'PENDING';
```

---

**Q: Show orders where delivery is overdue (not yet delivered but past delivery date)**
```sql
SELECT order_no, customer_name, order_status,
       delivery_date, grand_total
FROM v_sales_summary
WHERE delivery_date < CURRENT_DATE
AND order_status NOT IN ('DELIVERED', 'COMPLETED', 'VOID')
GROUP BY order_no, customer_name, order_status,
         delivery_date, grand_total
ORDER BY delivery_date ASC;
```

---

**Q: Show the top 10 orders by grand total this month**
```sql
SELECT order_no, customer_name, grand_total,
       total_quantity, order_status, delivery_method
FROM orders o
JOIN customer c ON o.customer_id = c.id
WHERE o.created_at >= date_trunc('month', CURRENT_DATE)
ORDER BY grand_total DESC
LIMIT 10;
```

---

**Q: Daily order count and revenue for the last 30 days**
```sql
SELECT order_date,
       COUNT(DISTINCT reference_no) AS order_count,
       SUM(quantity * unit_price)   AS revenue
FROM v_sales_summary
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY order_date
ORDER BY order_date ASC;
```

---

**Q: How many orders used delivery vs pickup this month?**
```sql
SELECT delivery_method,
       COUNT(DISTINCT order_no) AS order_count,
       SUM(grand_total)         AS total_value
FROM orders
WHERE created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY delivery_method
ORDER BY order_count DESC;
```

---

### SALES & REVENUE QUERIES

---

**Q: Total revenue this month**
```sql
SELECT SUM(quantity * unit_price)   AS revenue,
       COUNT(DISTINCT reference_no) AS order_count,
       SUM(quantity)                AS total_units_sold
FROM sales
WHERE order_date >= date_trunc('month', CURRENT_DATE);
```

---

**Q: Monthly revenue trend for this year**
```sql
SELECT date_trunc('month', order_date)  AS month,
       SUM(quantity * unit_price)       AS revenue,
       COUNT(DISTINCT reference_no)     AS order_count,
       SUM(quantity)                    AS units_sold
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Revenue by product this month**
```sql
SELECT product_name,
       product_code,
       SUM(quantity * unit_price)   AS revenue,
       SUM(quantity)                AS units_sold,
       COUNT(DISTINCT reference_no) AS order_count
FROM v_sales_summary
WHERE order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY product_name, product_code
ORDER BY revenue DESC
LIMIT 50;
```

---

**Q: Revenue by brand this year**
```sql
SELECT brand,
       SUM(quantity * unit_price)   AS revenue,
       SUM(quantity)                AS units_sold,
       COUNT(DISTINCT reference_no) AS order_count
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
AND brand IS NOT NULL
GROUP BY brand
ORDER BY revenue DESC;
```

---

**Q: Revenue by origin last month**
```sql
SELECT origin,
       SUM(quantity * unit_price) AS revenue,
       SUM(quantity)              AS units_sold
FROM sales
WHERE order_date BETWEEN
    date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
    AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day'
AND origin IS NOT NULL
GROUP BY origin
ORDER BY revenue DESC;
```

---

**Q: What is the average revenue per order this year?**
```sql
SELECT ROUND(
    SUM(quantity * unit_price)::NUMERIC / COUNT(DISTINCT reference_no), 2
) AS avg_revenue_per_order
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE);
```

---

**Q: Show weekly revenue for the last 12 weeks**
```sql
SELECT date_trunc('week', order_date)  AS week_start,
       SUM(quantity * unit_price)      AS revenue,
       COUNT(DISTINCT reference_no)    AS order_count
FROM sales
WHERE order_date >= CURRENT_DATE - INTERVAL '12 weeks'
GROUP BY week_start
ORDER BY week_start ASC;
```

---

**Q: Show sales lines that are still in PROCESSING status**
```sql
SELECT s.reference_no, s.customer_code, s.product_code,
       p.product_name, s.quantity, s.unit_price,
       s.order_date, s.sales_status
FROM sales s
JOIN product p ON s.product_code = p.product_code
WHERE s.sales_status = 'PROCESSING'
ORDER BY s.order_date ASC
LIMIT 100;
```

---

### PRODUCT QUERIES

---

**Q: Show all new arrival products**
```sql
SELECT product_code, product_name, product_number,
       status, created_at::date AS added_date
FROM product
WHERE new_arrival = TRUE
ORDER BY created_at DESC;
```

---

**Q: Show all unavailable products**
```sql
SELECT product_code, product_name, status, updated_at::date AS status_date
FROM product
WHERE status = 'OUT'
ORDER BY product_name ASC;
```

---

**Q: Which products have never been sold?**
```sql
SELECT p.product_code, p.product_name, p.status
FROM product p
LEFT JOIN sales s ON p.product_code = s.product_code
WHERE s.id IS NULL
ORDER BY p.product_name ASC;
```

---

**Q: Top 10 best selling products by quantity this year**
```sql
SELECT product_name,
       product_code,
       SUM(quantity)              AS total_units_sold,
       SUM(quantity * unit_price) AS revenue
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY product_name, product_code
ORDER BY total_units_sold DESC
LIMIT 10;
```

---

**Q: Top 10 best selling products by revenue last month**
```sql
SELECT product_name,
       product_code,
       SUM(quantity * unit_price) AS revenue,
       SUM(quantity)              AS total_units_sold
FROM v_sales_summary
WHERE order_date BETWEEN
    date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
    AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day'
GROUP BY product_name, product_code
ORDER BY revenue DESC
LIMIT 10;
```

---

**Q: Show product sales count by status (CALL vs OUT)**
```sql
SELECT p.status AS product_status,
       COUNT(DISTINCT p.product_code)  AS product_count,
       SUM(s.quantity)                 AS total_units_sold
FROM product p
LEFT JOIN sales s ON p.product_code = s.product_code
GROUP BY p.status
ORDER BY product_count DESC;
```

---

### INVENTORY QUERIES

---

**Q: Show current stock levels for all products**
```sql
SELECT p.product_code, p.product_name,
       i.on_hand, i.available_quantity,
       i.expected_quantity, i.sell_for, i.base_cost
FROM product p
LEFT JOIN inventory i ON p.product_code = i.product_code
ORDER BY i.available_quantity ASC NULLS FIRST;
```

---

**Q: Which products are low on stock?**
```sql
SELECT p.product_code, p.product_name,
       i.available_quantity, i.on_hand, i.expected_quantity
FROM product p
JOIN inventory i ON p.product_code = i.product_code
WHERE i.available_quantity < 50
ORDER BY i.available_quantity ASC;
```

---

**Q: Which products are completely out of stock?**
```sql
SELECT p.product_code, p.product_name,
       i.on_hand, i.available_quantity, i.expected_quantity
FROM product p
JOIN inventory i ON p.product_code = i.product_code
WHERE i.available_quantity = 0
ORDER BY p.product_name ASC;
```

---

**Q: What is the total stock value?**
```sql
SELECT SUM(i.on_hand * i.base_cost)          AS total_stock_cost,
       SUM(i.on_hand * i.sell_for)           AS total_stock_sell_value,
       SUM(i.available_quantity * i.base_cost) AS available_stock_cost
FROM inventory i;
```

---

**Q: Show stock value by product sorted highest to lowest**
```sql
SELECT p.product_name, p.product_code,
       i.on_hand,
       i.base_cost,
       i.sell_for,
       ROUND((i.on_hand * i.base_cost)::NUMERIC, 2) AS stock_cost_value,
       ROUND((i.on_hand * i.sell_for)::NUMERIC,  2) AS stock_sell_value
FROM product p
JOIN inventory i ON p.product_code = i.product_code
ORDER BY stock_cost_value DESC;
```

---

**Q: Show products with expected incoming stock**
```sql
SELECT p.product_code, p.product_name,
       i.on_hand, i.available_quantity,
       i.expected_quantity, i.receive_date
FROM product p
JOIN inventory i ON p.product_code = i.product_code
WHERE i.expected_quantity > 0
ORDER BY i.receive_date ASC;
```

---

**Q: Show inventory received this month**
```sql
SELECT p.product_name, p.product_code,
       i.lot_number, i.on_hand, i.available_quantity,
       i.base_cost, i.receive_date
FROM inventory i
JOIN product p ON i.product_code = p.product_code
WHERE i.receive_date >= date_trunc('month', CURRENT_DATE)
ORDER BY i.receive_date DESC;
```

---

**Q: Which products are fast moving (high sales vs low stock)?**
```sql
SELECT p.product_name,
       p.product_code,
       SUM(s.quantity)        AS units_sold_this_month,
       i.available_quantity   AS current_stock,
       CASE
           WHEN i.available_quantity = 0 THEN 'OUT OF STOCK'
           WHEN i.available_quantity < SUM(s.quantity) * 0.5 THEN 'CRITICAL'
           WHEN i.available_quantity < SUM(s.quantity) THEN 'LOW'
           ELSE 'ADEQUATE'
       END AS stock_health
FROM product p
JOIN inventory i ON p.product_code = i.product_code
JOIN sales s     ON p.product_code = s.product_code
WHERE s.order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY p.product_name, p.product_code, i.available_quantity
ORDER BY units_sold_this_month DESC;
```

---

### COMBINED / ANALYTICAL QUERIES

---

**Q: Revenue summary for this month vs last month**
```sql
SELECT
    SUM(CASE WHEN order_date >= date_trunc('month', CURRENT_DATE)
             THEN quantity * unit_price ELSE 0 END) AS revenue_this_month,
    SUM(CASE WHEN order_date BETWEEN
                 date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
                 AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day'
             THEN quantity * unit_price ELSE 0 END) AS revenue_last_month,
    COUNT(DISTINCT CASE WHEN order_date >= date_trunc('month', CURRENT_DATE)
                        THEN reference_no END) AS orders_this_month,
    COUNT(DISTINCT CASE WHEN order_date BETWEEN
                 date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
                 AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day'
                        THEN reference_no END) AS orders_last_month
FROM v_sales_summary;
```

---

**Q: Show customer order history for customer code CUST001**
```sql
SELECT order_no, order_status, grand_total,
       delivery_method, delivery_date,
       created_at::date AS order_date
FROM v_sales_summary
WHERE customer_code = 'CUST001'
GROUP BY order_no, order_status, grand_total,
         delivery_method, delivery_date, created_at::date
ORDER BY created_at DESC;
```

---

**Q: Which customers have orders over 10000 in grand total this year?**
```sql
SELECT customer_name, customer_code,
       COUNT(DISTINCT order_no) AS order_count,
       SUM(grand_total)         AS total_spend
FROM v_sales_summary
WHERE created_at >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name, customer_code
HAVING SUM(grand_total) > 10000
ORDER BY total_spend DESC;
```

---

**Q: Show product performance — revenue, units sold, and current stock**
```sql
SELECT v.product_name,
       v.product_code,
       SUM(v.quantity * v.unit_price) AS revenue,
       SUM(v.quantity)                AS units_sold,
       MAX(v.available_quantity)      AS current_stock,
       MAX(v.on_hand)                 AS on_hand
FROM v_sales_summary v
WHERE v.order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY v.product_name, v.product_code
ORDER BY revenue DESC
LIMIT 50;
```

---

**Q: How many distinct customers ordered each product this year?**
```sql
SELECT p.product_name, p.product_code,
       COUNT(DISTINCT s.customer_code) AS unique_customers,
       SUM(s.quantity)                 AS total_units_sold
FROM sales s
JOIN product p ON s.product_code = p.product_code
WHERE s.order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY p.product_name, p.product_code
ORDER BY unique_customers DESC;
```

---

**Q: Show daily revenue for the current month so far**
```sql
SELECT order_date,
       SUM(quantity * unit_price)   AS daily_revenue,
       COUNT(DISTINCT reference_no) AS orders,
       SUM(quantity)                AS units_sold
FROM sales
WHERE order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY order_date
ORDER BY order_date ASC;
```

---

**Q: Which customers placed the most orders this year?**
```sql
SELECT customer_name, customer_code,
       COUNT(DISTINCT reference_no) AS order_count,
       SUM(quantity * unit_price)   AS total_revenue
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name, customer_code
ORDER BY order_count DESC
LIMIT 20;
```

---

**Q: Show me a full summary — total revenue, orders, customers, and units for this month**
```sql
SELECT
    SUM(quantity * unit_price)      AS total_revenue,
    COUNT(DISTINCT reference_no)    AS total_orders,
    COUNT(DISTINCT customer_code)   AS unique_customers,
    SUM(quantity)                   AS total_units_sold,
    COUNT(DISTINCT product_code)    AS unique_products_sold,
    ROUND(
        SUM(quantity * unit_price)::NUMERIC /
        NULLIF(COUNT(DISTINCT reference_no), 0), 2
    ) AS avg_order_value
FROM v_sales_summary
WHERE order_date >= date_trunc('month', CURRENT_DATE);
```

---

---

## VISUALIZATION-TARGETED EXAMPLE QUERIES

---

### LINE CHART QUERIES — date + single metric (2 columns)

---

**Q: Show monthly revenue trend for this year (line chart)**
```sql
SELECT date_trunc('month', order_date)::date AS month,
       SUM(quantity * unit_price)            AS revenue
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show daily order count for the last 30 days (line chart)**
```sql
SELECT created_at::date             AS order_date,
       COUNT(DISTINCT order_no)     AS order_count
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY order_date
ORDER BY order_date ASC;
```

---

**Q: Show weekly revenue for the last 12 weeks (line chart)**
```sql
SELECT date_trunc('week', order_date)::date AS week_start,
       SUM(quantity * unit_price)           AS revenue
FROM sales
WHERE order_date >= CURRENT_DATE - INTERVAL '12 weeks'
GROUP BY week_start
ORDER BY week_start ASC;
```

---

**Q: Show monthly new customer registrations this year (line chart)**
```sql
SELECT date_trunc('month', created_at)::date AS month,
       COUNT(*)                              AS new_customers
FROM customer
WHERE created_at >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show upcoming expected stock arrivals by date (line chart)**
```sql
SELECT receive_date,
       SUM(expected_quantity) AS expected_units
FROM inventory
WHERE receive_date >= CURRENT_DATE
  AND expected_quantity > 0
GROUP BY receive_date
ORDER BY receive_date ASC;
```

---

**Q: Show daily units sold for the current month (line chart)**
```sql
SELECT order_date,
       SUM(quantity) AS units_sold
FROM sales
WHERE order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY order_date
ORDER BY order_date ASC;
```

---

**Q: Show monthly order count trend this year (line chart)**
```sql
SELECT date_trunc('month', order_date)::date AS month,
       COUNT(DISTINCT reference_no)          AS order_count
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

### BAR CHART QUERIES — category + single metric, LIMIT ≤ 20 (2 columns)

---

**Q: Top 10 customers by revenue this year (bar chart)**
```sql
SELECT customer_name,
       SUM(quantity * unit_price) AS revenue
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name
ORDER BY revenue DESC
LIMIT 10;
```

---

**Q: Revenue by brand this year (bar chart)**
```sql
SELECT brand,
       SUM(quantity * unit_price) AS revenue
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
  AND brand IS NOT NULL
GROUP BY brand
ORDER BY revenue DESC
LIMIT 15;
```

---

**Q: Top 10 products by units sold this month (bar chart)**
```sql
SELECT product_name,
       SUM(quantity) AS units_sold
FROM v_sales_summary
WHERE order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY product_name
ORDER BY units_sold DESC
LIMIT 10;
```

---

**Q: Order count by status this month (bar chart)**
```sql
SELECT order_status,
       COUNT(DISTINCT order_no) AS order_count
FROM orders
WHERE created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY order_status
ORDER BY order_count DESC;
```

---

**Q: Top 10 products with highest on-hand stock (bar chart)**
```sql
SELECT p.product_name,
       i.on_hand AS stock_quantity
FROM product p
JOIN inventory i ON p.product_code = i.product_code
WHERE i.on_hand > 0
ORDER BY i.on_hand DESC
LIMIT 10;
```

---

**Q: Revenue by origin this year (bar chart)**
```sql
SELECT COALESCE(origin, 'Unknown') AS origin,
       SUM(quantity * unit_price)  AS revenue
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY origin
ORDER BY revenue DESC
LIMIT 15;
```

---

**Q: Top 10 customers by order count this year (bar chart)**
```sql
SELECT customer_name,
       COUNT(DISTINCT reference_no) AS order_count
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name
ORDER BY order_count DESC
LIMIT 10;
```

---

**Q: Top 10 products by revenue last month (bar chart)**
```sql
SELECT product_name,
       SUM(quantity * unit_price) AS revenue
FROM v_sales_summary
WHERE order_date BETWEEN
    date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
    AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day'
GROUP BY product_name
ORDER BY revenue DESC
LIMIT 10;
```

---

### PIE CHART QUERIES — category + percentage/share/proportion, LIMIT ≤ 8 (2 columns)

---

**Q: What is the percentage of active vs inactive customers? (pie chart)**
```sql
SELECT activation_status,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM customer
GROUP BY activation_status
ORDER BY percentage DESC;
```

---

**Q: Show customer credit status distribution (pie chart)**
```sql
SELECT credit_status,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM customer
GROUP BY credit_status
ORDER BY percentage DESC;
```

---

**Q: What percentage of orders use delivery vs pickup this year? (pie chart)**
```sql
SELECT delivery_method,
       ROUND(COUNT(DISTINCT order_no) * 100.0 / SUM(COUNT(DISTINCT order_no)) OVER (), 1) AS percentage
FROM orders
WHERE created_at >= date_trunc('year', CURRENT_DATE)
GROUP BY delivery_method
ORDER BY percentage DESC;
```

---

**Q: Show order status share this month (pie chart)**
```sql
SELECT order_status,
       ROUND(COUNT(DISTINCT order_no) * 100.0 / SUM(COUNT(DISTINCT order_no)) OVER (), 1) AS percentage
FROM orders
WHERE created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY order_status
ORDER BY percentage DESC
LIMIT 8;
```

---

**Q: Revenue share by brand this year (pie chart)**
```sql
SELECT brand,
       ROUND(SUM(quantity * unit_price)::NUMERIC * 100.0 / SUM(SUM(quantity * unit_price)::NUMERIC) OVER (), 1) AS revenue_share
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
  AND brand IS NOT NULL
GROUP BY brand
ORDER BY revenue_share DESC
LIMIT 8;
```

---

**Q: Show product availability status distribution (pie chart)**
```sql
SELECT status AS product_status,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM product
GROUP BY status
ORDER BY percentage DESC;
```

---

**Q: Show inventory stock health distribution (pie chart)**
```sql
SELECT
    CASE
        WHEN available_quantity = 0    THEN 'Out of Stock'
        WHEN available_quantity < 50   THEN 'Low Stock'
        ELSE                                'Adequate'
    END                                              AS stock_category,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM inventory
GROUP BY stock_category
ORDER BY percentage DESC;
```

---

**Q: Revenue share by origin this year (pie chart)**
```sql
SELECT COALESCE(origin, 'Unknown') AS origin,
       ROUND(SUM(quantity * unit_price)::NUMERIC * 100.0 / SUM(SUM(quantity * unit_price)::NUMERIC) OVER (), 1) AS revenue_share
FROM sales
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY origin
ORDER BY revenue_share DESC
LIMIT 8;
```

---

**Q: Sales line status share (pie chart)**
```sql
SELECT sales_status,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM sales
GROUP BY sales_status
ORDER BY percentage DESC;
```

---

### ADDITIONAL COMBINED / ANALYTICAL QUERIES

---

**Q: Monthly revenue vs order count side-by-side for this year**
```sql
SELECT date_trunc('month', order_date)::date AS month,
       SUM(quantity * unit_price)            AS revenue,
       COUNT(DISTINCT reference_no)          AS order_count,
       SUM(quantity)                         AS units_sold
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Top 10 customers with their order count, revenue, and units this year**
```sql
SELECT customer_name,
       COUNT(DISTINCT reference_no) AS order_count,
       SUM(quantity * unit_price)   AS revenue,
       SUM(quantity)                AS total_units
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY customer_name
ORDER BY revenue DESC
LIMIT 10;
```

---

**Q: Show stock vs sales comparison for top 10 products this month**
```sql
SELECT v.product_name,
       SUM(v.quantity)           AS units_sold,
       MAX(v.available_quantity) AS available_stock,
       MAX(v.on_hand)            AS on_hand
FROM v_sales_summary v
WHERE v.order_date >= date_trunc('month', CURRENT_DATE)
GROUP BY v.product_name
ORDER BY units_sold DESC
LIMIT 10;
```

---

**Q: Show customer retention — first order date, last order date, and total orders per customer**
```sql
SELECT customer_name,
       MIN(order_date)              AS first_order_date,
       MAX(order_date)              AS last_order_date,
       COUNT(DISTINCT reference_no) AS total_orders,
       SUM(quantity * unit_price)   AS total_revenue
FROM v_sales_summary
GROUP BY customer_name
ORDER BY total_revenue DESC
LIMIT 50;
```

---

**Q: Delivery performance — on-time vs overdue orders this year**
```sql
SELECT
    SUM(CASE WHEN delivery_date >= order_date THEN 1 ELSE 0 END) AS on_time_orders,
    SUM(CASE WHEN delivery_date < order_date  THEN 1 ELSE 0 END) AS overdue_orders,
    COUNT(DISTINCT order_no)                                      AS total_orders
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
  AND order_status NOT IN ('VOID', 'PENDING');
```

---

**Q: Show gross profit by product this year (revenue minus cost)**
```sql
SELECT product_name,
       product_code,
       SUM(quantity * unit_price)             AS revenue,
       SUM(quantity * base_cost)              AS total_cost,
       SUM((unit_price - base_cost) * quantity) AS gross_profit
FROM v_sales_summary
WHERE order_date >= date_trunc('year', CURRENT_DATE)
GROUP BY product_name, product_code
ORDER BY gross_profit DESC
LIMIT 20;
```

---

**Q: Show month-over-month revenue growth this year**
```sql
WITH monthly AS (
    SELECT date_trunc('month', order_date)::date AS month,
           SUM(quantity * unit_price)::NUMERIC   AS revenue
    FROM sales
    WHERE order_date >= date_trunc('year', CURRENT_DATE)
    GROUP BY month
)
SELECT month,
       revenue,
       LAG(revenue) OVER (ORDER BY month)  AS prev_month_revenue,
       ROUND(
           (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0
           / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1
       ) AS growth_percentage
FROM monthly
ORDER BY month ASC;
```

---

**Q: Show the top 5 products ordered by customers on credit hold**
```sql
SELECT v.product_name,
       v.product_code,
       COUNT(DISTINCT v.reference_no) AS order_count,
       SUM(v.quantity)                AS total_units
FROM v_sales_summary v
WHERE v.credit_status = 'HOLD'
GROUP BY v.product_name, v.product_code
ORDER BY order_count DESC
LIMIT 5;
```

---

## END OF EXAMPLE QUERY PAIRS