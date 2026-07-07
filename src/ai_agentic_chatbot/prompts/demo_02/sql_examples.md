## EXAMPLE QUERY PAIRS

---

### BRAND & SUB-BRAND QUERIES

---

**Q: List all brands**
```sql
SELECT b.brand_code, b.brand_name
FROM brand_mst b
ORDER BY b.brand_name ASC;
```

---

**Q: What is the brand name for code BR001?**
```sql
SELECT b.brand_name
FROM brand_mst b
WHERE b.brand_code = 'BR001';
```

---

**Q: List all sub-brands**
```sql
SELECT sb.sub_brand_code, sb.sub_brand_name
FROM sub_brand_mst sb
ORDER BY sb.sub_brand_name ASC;
```

---

**Q: How many brands do we have?**
```sql
SELECT COUNT(*) AS brand_count
FROM brand_mst b;
```

---

**Q: Search for brands with "cotton" in the name**
```sql
SELECT b.brand_code, b.brand_name
FROM brand_mst b
WHERE b.brand_name ILIKE '%cotton%'
ORDER BY b.brand_name ASC;
```

---

**Q: Find the sub-brand code for sub-brand name "Premium Line"**
```sql
SELECT sb.sub_brand_code, sb.sub_brand_name
FROM sub_brand_mst sb
WHERE sb.sub_brand_name ILIKE '%Premium Line%';
```

---

### REGION QUERIES

---

**Q: List all regions**
```sql
SELECT r.region_code, r.region_name
FROM region_mst r
ORDER BY r.region_name ASC;
```

---

**Q: What is the region name for code 5?**
```sql
SELECT r.region_name
FROM region_mst r
WHERE r.region_code = 5;
```

---

**Q: How many regions do we have?**
```sql
SELECT COUNT(*) AS region_count
FROM region_mst r;
```

---

**Q: Show total sales amount by region this year**
```sql
SELECT sc.region_code,
       sc.region_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY sc.region_code, sc.region_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Validate sales data against the region master — show sales amount joined to the official region name**
```sql
SELECT r.region_code,
       r.region_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
JOIN region_mst r ON sc.region_code::bigint = r.region_code
GROUP BY r.region_code, r.region_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Which regions appearing in the sales data have no matching entry in the region master?**
```sql
SELECT DISTINCT sc.region_code, sc.region_name
FROM sales_comparison sc
LEFT JOIN region_mst r ON sc.region_code::bigint = r.region_code
WHERE r.id IS NULL
ORDER BY sc.region_name ASC;
```

---

### CUSTOMER QUERIES

---

**Q: List all customers**
```sql
SELECT c."Customer", c."Name 1", c."City"
FROM customer_mst c
ORDER BY c."Name 1" ASC
LIMIT 100;
```

---

**Q: Which customers are located in Mumbai?**
```sql
SELECT c."Customer", c."Name 1", c."City"
FROM customer_mst c
WHERE c."City" ILIKE '%Mumbai%'
ORDER BY c."Name 1" ASC;
```

---

**Q: How many customers are in each city?**
```sql
SELECT c."City" AS city,
       COUNT(*) AS customer_count
FROM customer_mst c
GROUP BY c."City"
ORDER BY customer_count DESC
LIMIT 100;
```

---

**Q: Find the customer with ID CUST123**
```sql
SELECT c."Customer", c."Name 1", c."City"
FROM customer_mst c
WHERE c."Customer" = 'CUST123';
```

---

**Q: Show top 10 customers by sales amount this year**
```sql
SELECT sc.customer_code,
       sc.customer_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY sc.customer_code, sc.customer_name
ORDER BY sales_amount DESC
LIMIT 10;
```

---

**Q: Show sales amount for customer code CUST123, validated against the official customer master name**
```sql
SELECT c."Customer"   AS customer_code,
       c."Name 1"     AS customer_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
JOIN customer_mst c ON sc.customer_code = c."Customer"
WHERE sc.customer_code = 'CUST123'
GROUP BY c."Customer", c."Name 1";
```

---

**Q: How many distinct customers appear in the sales comparison data?**
```sql
SELECT COUNT(DISTINCT sc.customer_code) AS distinct_customer_count
FROM sales_comparison sc;
```

---

### CONSUMPTION QUERIES

---

**Q: Show total quantity consumed by supplier**
```sql
SELECT cn.name1 AS supplier_name,
       SUM(cn.erfmg) AS quantity_consumed
FROM consumption cn
GROUP BY cn.name1
ORDER BY quantity_consumed DESC
LIMIT 100;
```

---

**Q: Show total waste by material description**
```sql
SELECT cn.maktx AS material_description,
       SUM(cn.waste) AS total_waste
FROM consumption cn
GROUP BY cn.maktx
ORDER BY total_waste DESC
LIMIT 100;
```

---

**Q: Which purchase orders have the highest consumption variance?**
```sql
SELECT cn.ebeln, cn.name1 AS supplier_name, cn.diff AS consumption_variance
FROM consumption cn
WHERE cn.diff IS NOT NULL
ORDER BY cn.diff DESC
LIMIT 20;
```

---

**Q: Show total cut quantity by finished goods material**
```sql
SELECT cn.fg AS finished_goods,
       SUM(cn.cut_qty::numeric) AS total_cut_qty
FROM consumption cn
WHERE cn.cut_qty IS NOT NULL
GROUP BY cn.fg
ORDER BY total_cut_qty DESC
LIMIT 100;
```

---

**Q: What is the average consumption percentage across all entries?**
```sql
SELECT ROUND(AVG(cn.perc)::NUMERIC, 2) AS avg_consumption_percentage
FROM consumption cn;
```

---

**Q: Show consumption entries where waste exceeded 100**
```sql
SELECT cn.charg AS batch_number, cn.matnr_fg, cn.name1 AS supplier_name, cn.waste
FROM consumption cn
WHERE cn.waste > 100
ORDER BY cn.waste DESC
LIMIT 100;
```

---

**Q: Show total quantity consumed (menge) by batch**
```sql
SELECT cn.charg AS batch_number,
       SUM(cn.menge::numeric) AS total_menge
FROM consumption cn
WHERE cn.menge IS NOT NULL
GROUP BY cn.charg
ORDER BY total_menge DESC
LIMIT 100;
```

---

**Q: Which suppliers have the highest cumulative waste?**
```sql
SELECT cn.name1 AS supplier_name,
       SUM(cn.wast) AS cumulative_waste
FROM consumption cn
GROUP BY cn.name1
ORDER BY cumulative_waste DESC
LIMIT 10;
```

---

### SALES COMPARISON QUERIES

---

**Q: Show total sales amount by brand this year**
```sql
SELECT sc.brand_code, sc.brand_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY sc.brand_code, sc.brand_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Show total sales quantity by sub-brand**
```sql
SELECT sc.sub_brand_code, sc.sub_brand_name,
       SUM(sc.qun1_r) AS sales_quantity
FROM sales_comparison sc
GROUP BY sc.sub_brand_code, sc.sub_brand_name
ORDER BY sales_quantity DESC
LIMIT 100;
```

---

**Q: Which customers have the highest planned amount for period 1?**
```sql
SELECT sc.customer_code, sc.customer_name,
       SUM(sc.pamt1) AS planned_amount
FROM sales_comparison sc
GROUP BY sc.customer_code, sc.customer_name
ORDER BY planned_amount DESC
LIMIT 20;
```

---

**Q: Show total sales amount for item code ITM001**
```sql
SELECT sc.item_code, sc.item_name,
       SUM(sc.amount1_r) AS sales_amount,
       SUM(sc.qun1_r)    AS sales_quantity
FROM sales_comparison sc
WHERE sc.item_code = 'ITM001'
GROUP BY sc.item_code, sc.item_name;
```

---

**Q: Compare planned vs actual sales quantity by sub-brand**
```sql
SELECT sc.sub_brand_code, sc.sub_brand_name,
       SUM(sc.pqty1)  AS planned_quantity,
       SUM(sc.qun1_r) AS actual_quantity
FROM sales_comparison sc
GROUP BY sc.sub_brand_code, sc.sub_brand_name
ORDER BY actual_quantity DESC
LIMIT 100;
```

---

**Q: Show sales records where actual quantity sold is zero but planned quantity was greater than zero**
```sql
SELECT sc.item_code, sc.item_name, sc.customer_name,
       sc.pqty1 AS planned_quantity, sc.qun1_r AS actual_quantity
FROM sales_comparison sc
WHERE COALESCE(sc.qun1_r, 0) = 0
  AND COALESCE(sc.pqty1, 0) > 0
ORDER BY sc.pqty1 DESC
LIMIT 100;
```

---

**Q: What is the total sales amount recorded this month?**
```sql
SELECT SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('month', CURRENT_DATE);
```

---

### COMBINED / ANALYTICAL QUERIES

---

**Q: Show sales amount by brand validated against the canonical brand master names**
```sql
SELECT b.brand_code, b.brand_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
JOIN brand_mst b ON sc.brand_code = b.brand_code
GROUP BY b.brand_code, b.brand_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Show sales performance by region with official region master names**
```sql
SELECT r.region_code, r.region_name,
       SUM(sc.amount1_r) AS sales_amount,
       SUM(sc.qun1_r)    AS sales_quantity
FROM sales_comparison sc
JOIN region_mst r ON sc.region_code::bigint = r.region_code
GROUP BY r.region_code, r.region_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Show top 10 customers by sales amount along with their registered city from the customer master**
```sql
SELECT c."Customer" AS customer_code,
       c."Name 1"   AS customer_name,
       c."City"     AS city,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
JOIN customer_mst c ON sc.customer_code = c."Customer"
GROUP BY c."Customer", c."Name 1", c."City"
ORDER BY sales_amount DESC
LIMIT 10;
```

---

**Q: Show total waste and total cut quantity by supplier**
```sql
SELECT cn.name1 AS supplier_name,
       SUM(cn.waste) AS total_waste,
       SUM(cn.cut_qty::numeric) AS total_cut_qty
FROM consumption cn
GROUP BY cn.name1
ORDER BY total_waste DESC
LIMIT 100;
```

---

**Q: Show sales amount and sales quantity by brand and region together**
```sql
SELECT sc.brand_name, sc.region_name,
       SUM(sc.amount1_r) AS sales_amount,
       SUM(sc.qun1_r)    AS sales_quantity
FROM sales_comparison sc
GROUP BY sc.brand_name, sc.region_name
ORDER BY sales_amount DESC
LIMIT 100;
```

---

**Q: Compare total sales amount to total planned amount by brand this year**
```sql
SELECT sc.brand_code, sc.brand_name,
       SUM(sc.amount1_r) AS actual_sales_amount,
       SUM(sc.pamt1)     AS planned_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY sc.brand_code, sc.brand_name
ORDER BY actual_sales_amount DESC
LIMIT 100;
```

---

**Q: Show sub-brands validated against the master along with total planned quantity**
```sql
SELECT sb.sub_brand_code, sb.sub_brand_name,
       SUM(sc.pqty1) AS planned_quantity
FROM sales_comparison sc
JOIN sub_brand_mst sb ON sc.sub_brand_code = sb.sub_brand_code
GROUP BY sb.sub_brand_code, sb.sub_brand_name
ORDER BY planned_quantity DESC
LIMIT 100;
```

---

## VISUALIZATION-TARGETED EXAMPLE QUERIES

---

### LINE CHART QUERIES — date + single metric (2 columns)

---

**Q: Show monthly sales amount trend for this year (line chart)**
```sql
SELECT date_trunc('month', to_date(sc.fkdat, 'YYYYMMDD'))::date AS month,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show monthly sales quantity trend for this year (line chart)**
```sql
SELECT date_trunc('month', to_date(sc.fkdat, 'YYYYMMDD'))::date AS month,
       SUM(sc.qun1_r) AS sales_quantity
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show daily quantity consumed for the last 90 days (line chart)**
```sql
SELECT to_date(cn.bedat, 'YYYYMMDD') AS consumption_date,
       SUM(cn.erfmg) AS quantity_consumed
FROM consumption cn
WHERE to_date(cn.bedat, 'YYYYMMDD') >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY consumption_date
ORDER BY consumption_date ASC;
```

---

**Q: Show daily waste trend for the last 30 days (line chart)**
```sql
SELECT to_date(cn.bedat, 'YYYYMMDD') AS consumption_date,
       SUM(cn.waste) AS total_waste
FROM consumption cn
WHERE to_date(cn.bedat, 'YYYYMMDD') >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY consumption_date
ORDER BY consumption_date ASC;
```

---

**Q: Show monthly consumption variance trend for this year (line chart)**
```sql
SELECT date_trunc('month', to_date(cn.bedat, 'YYYYMMDD'))::date AS month,
       SUM(cn.diff) AS consumption_variance
FROM consumption cn
WHERE to_date(cn.bedat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show weekly planned amount trend for this year (line chart)**
```sql
SELECT date_trunc('week', to_date(sc.fkdat, 'YYYYMMDD'))::date AS week_start,
       SUM(sc.pamt1) AS planned_amount
FROM sales_comparison sc
WHERE to_date(sc.fkdat, 'YYYYMMDD') >= date_trunc('year', CURRENT_DATE)
GROUP BY week_start
ORDER BY week_start ASC;
```

---

### BAR CHART QUERIES — category + single metric, LIMIT ≤ 20 (2 columns)

---

**Q: Top 10 brands by sales amount (bar chart)**
```sql
SELECT sc.brand_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
GROUP BY sc.brand_name
ORDER BY sales_amount DESC
LIMIT 10;
```

---

**Q: Top 10 customers by sales amount (bar chart)**
```sql
SELECT sc.customer_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
GROUP BY sc.customer_name
ORDER BY sales_amount DESC
LIMIT 10;
```

---

**Q: Top 10 regions by sales quantity (bar chart)**
```sql
SELECT sc.region_name,
       SUM(sc.qun1_r) AS sales_quantity
FROM sales_comparison sc
GROUP BY sc.region_name
ORDER BY sales_quantity DESC
LIMIT 10;
```

---

**Q: Top 10 items by sales amount (bar chart)**
```sql
SELECT sc.item_name,
       SUM(sc.amount1_r) AS sales_amount
FROM sales_comparison sc
GROUP BY sc.item_name
ORDER BY sales_amount DESC
LIMIT 10;
```

---

**Q: Top 10 suppliers by total waste (bar chart)**
```sql
SELECT cn.name1 AS supplier_name,
       SUM(cn.waste) AS total_waste
FROM consumption cn
GROUP BY cn.name1
ORDER BY total_waste DESC
LIMIT 10;
```

---

**Q: Top 10 sub-brands by planned amount (bar chart)**
```sql
SELECT sc.sub_brand_name,
       SUM(sc.pamt1) AS planned_amount
FROM sales_comparison sc
GROUP BY sc.sub_brand_name
ORDER BY planned_amount DESC
LIMIT 10;
```

---

**Q: Top 10 materials by quantity consumed (bar chart)**
```sql
SELECT cn.maktx AS material_description,
       SUM(cn.erfmg) AS quantity_consumed
FROM consumption cn
GROUP BY cn.maktx
ORDER BY quantity_consumed DESC
LIMIT 10;
```

---

### PIE CHART QUERIES — category + percentage/share/proportion, LIMIT ≤ 8 (2 columns)

---

**Q: Show sales amount share by brand (pie chart)**
```sql
SELECT sc.brand_name,
       ROUND(SUM(sc.amount1_r)::NUMERIC * 100.0 / SUM(SUM(sc.amount1_r)::NUMERIC) OVER (), 1) AS sales_share
FROM sales_comparison sc
GROUP BY sc.brand_name
ORDER BY sales_share DESC
LIMIT 8;
```

---

**Q: Show sales quantity share by region (pie chart)**
```sql
SELECT sc.region_name,
       ROUND(SUM(sc.qun1_r)::NUMERIC * 100.0 / SUM(SUM(sc.qun1_r)::NUMERIC) OVER (), 1) AS sales_share
FROM sales_comparison sc
GROUP BY sc.region_name
ORDER BY sales_share DESC
LIMIT 8;
```

---

**Q: Show waste share by supplier (pie chart)**
```sql
SELECT cn.name1 AS supplier_name,
       ROUND(SUM(cn.waste)::NUMERIC * 100.0 / SUM(SUM(cn.waste)::NUMERIC) OVER (), 1) AS waste_share
FROM consumption cn
GROUP BY cn.name1
ORDER BY waste_share DESC
LIMIT 8;
```

---

**Q: Show sales amount share by sub-brand (pie chart)**
```sql
SELECT sc.sub_brand_name,
       ROUND(SUM(sc.amount1_r)::NUMERIC * 100.0 / SUM(SUM(sc.amount1_r)::NUMERIC) OVER (), 1) AS sales_share
FROM sales_comparison sc
GROUP BY sc.sub_brand_name
ORDER BY sales_share DESC
LIMIT 8;
```

---

**Q: Show planned amount proportion by customer, top 8 (pie chart)**
```sql
SELECT sc.customer_name,
       ROUND(SUM(sc.pamt1)::NUMERIC * 100.0 / SUM(SUM(sc.pamt1)::NUMERIC) OVER (), 1) AS planned_amount_proportion
FROM sales_comparison sc
GROUP BY sc.customer_name
ORDER BY planned_amount_proportion DESC
LIMIT 8;
```

---

**Q: Show consumed quantity share by finished goods material, top 8 (pie chart)**
```sql
SELECT cn.fg AS finished_goods,
       ROUND(SUM(cn.erfmg)::NUMERIC * 100.0 / SUM(SUM(cn.erfmg)::NUMERIC) OVER (), 1) AS quantity_share
FROM consumption cn
GROUP BY cn.fg
ORDER BY quantity_share DESC
LIMIT 8;
```

---

## END OF EXAMPLE QUERY PAIRS