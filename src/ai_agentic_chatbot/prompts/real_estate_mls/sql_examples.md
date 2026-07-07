## EXAMPLE QUERY PAIRS

---

### OFFICE QUERIES

---

**Q: List all real estate offices**
```sql
SELECT o.office_id, o.office_name, o.brokerage_name, o.region, o.market_area
FROM offices o
ORDER BY o.office_name ASC;
```

---

**Q: Which brokerage is a specific office affiliated with?**
```sql
SELECT o.office_name, o.brokerage_name
FROM offices o
WHERE o.office_name ILIKE '%Downtown%';
```

---

**Q: How many offices do we have in each region?**
```sql
SELECT o.region, COUNT(*) AS office_count
FROM offices o
GROUP BY o.region
ORDER BY office_count DESC;
```

---

**Q: List offices in the Northeast market area**
```sql
SELECT o.office_id, o.office_name, o.brokerage_name, o.market_area
FROM offices o
WHERE o.market_area ILIKE '%Northeast%'
ORDER BY o.office_name ASC;
```

---

### AGENT QUERIES

---

**Q: Who are the active agents?**
```sql
SELECT a.agent_id, a.agent_name, a.license_number, a.hire_date
FROM agents a
WHERE a.is_active = TRUE
ORDER BY a.agent_name ASC
LIMIT 100;
```

---

**Q: Which office does a specific agent work for?**
```sql
SELECT a.agent_name, o.office_name, o.brokerage_name
FROM agents a
LEFT JOIN offices o ON a.office_id = o.office_id
WHERE a.agent_name ILIKE '%Smith%';
```

---

**Q: How many agents work at each office?**
```sql
SELECT o.office_name, COUNT(a.agent_id) AS agent_count
FROM offices o
LEFT JOIN agents a ON o.office_id = a.office_id
GROUP BY o.office_name
ORDER BY agent_count DESC;
```

---

**Q: Which agents have no office assigned?**
```sql
SELECT a.agent_id, a.agent_name, a.license_number
FROM agents a
WHERE a.office_id IS NULL
ORDER BY a.agent_name ASC;
```

---

**Q: How long has each agent been with the company (tenure in years)?**
```sql
SELECT a.agent_name,
       a.hire_date,
       EXTRACT(YEAR FROM AGE(CURRENT_DATE, a.hire_date)) AS years_of_tenure
FROM agents a
WHERE a.is_active = TRUE
ORDER BY years_of_tenure DESC
LIMIT 50;
```

---

### PROPERTY QUERIES

---

**Q: What properties are available in a specific city?**
```sql
SELECT p.property_id, p.address_line1, p.city, p.state,
       p.property_type, p.bedrooms, p.bathrooms, p.square_footage
FROM properties p
WHERE p.city ILIKE '%Austin%'
ORDER BY p.address_line1 ASC
LIMIT 100;
```

---

**Q: How many bedrooms and bathrooms does a specific property have?**
```sql
SELECT p.address_line1, p.city, p.bedrooms, p.bathrooms
FROM properties p
WHERE p.address_line1 ILIKE '%Main St%';
```

---

**Q: Show properties with a homeowners association fee**
```sql
SELECT p.address_line1, p.city, p.state, p.hoa_fee
FROM properties p
WHERE p.hoa_flag = TRUE
ORDER BY p.hoa_fee DESC
LIMIT 100;
```

---

**Q: What is the average square footage of properties by property type?**
```sql
SELECT p.property_type,
       ROUND(AVG(p.square_footage)::NUMERIC, 0) AS avg_square_footage,
       COUNT(*) AS property_count
FROM properties p
GROUP BY p.property_type
ORDER BY avg_square_footage DESC;
```

---

**Q: Which properties were built before 1980?**
```sql
SELECT p.address_line1, p.city, p.state, p.year_built
FROM properties p
WHERE p.year_built < 1980
ORDER BY p.year_built ASC
LIMIT 100;
```

---

**Q: Show properties by zoning type**
```sql
SELECT p.zoning, COUNT(*) AS property_count
FROM properties p
WHERE p.zoning IS NOT NULL
GROUP BY p.zoning
ORDER BY property_count DESC;
```

---

### LISTING QUERIES

---

**Q: What are the current active listings?**
```sql
SELECT vld.listing_id, vld.address_line1, vld.city, vld.state,
       vld.list_price, vld.date_listed, vld.agent_name
FROM vw_listing_details vld
WHERE vld.status = 'ACTIVE'
ORDER BY vld.date_listed DESC
LIMIT 100;
```

---

**Q: How long has a specific property been on the market?**
```sql
SELECT vld.address_line1, vld.city, vld.status, vld.days_on_market
FROM vw_listing_details vld
WHERE vld.address_line1 ILIKE '%Oak Ave%';
```

---

**Q: What was the original list price of a property compared to its current list price?**
```sql
SELECT vld.address_line1, vld.city,
       vld.original_list_price, vld.list_price
FROM vw_listing_details vld
WHERE vld.address_line1 ILIKE '%Elm St%';
```

---

**Q: How many listings closed above their list price?**
```sql
SELECT COUNT(*) AS above_list_price_count
FROM listings l
WHERE l.status = 'CLOSED'
  AND l.sale_price > l.list_price;
```

---

**Q: What is the average commission percentage on active listings?**
```sql
SELECT ROUND(AVG(l.commission_pct)::NUMERIC, 2) AS avg_commission_pct
FROM listings l
WHERE l.status = 'ACTIVE';
```

---

**Q: Show listings that expired without selling**
```sql
SELECT vld.address_line1, vld.city, vld.list_price,
       vld.date_listed, vld.date_expired
FROM vw_listing_details vld
WHERE vld.date_expired IS NOT NULL
ORDER BY vld.date_expired DESC
LIMIT 100;
```

---

**Q: What are the different listing statuses in the system?**
```sql
SELECT DISTINCT l.status
FROM listings l
ORDER BY l.status ASC;
```

---

**Q: Show the top 10 most expensive active listings**
```sql
SELECT vld.address_line1, vld.city, vld.state, vld.list_price
FROM vw_listing_details vld
WHERE vld.status = 'ACTIVE'
ORDER BY vld.list_price DESC
LIMIT 10;
```

---

**Q: How many listings came from each source?**
```sql
SELECT l.source, COUNT(*) AS listing_count
FROM listings l
WHERE l.source IS NOT NULL
GROUP BY l.source
ORDER BY listing_count DESC;
```

---

### PRICE HISTORY QUERIES

---

**Q: What were the previous prices for a specific listing?**
```sql
SELECT ph.change_date, ph.old_price, ph.new_price
FROM price_history ph
WHERE ph.listing_id = 101
ORDER BY ph.change_date ASC;
```

---

**Q: When did the price last change for a property?**
```sql
SELECT vld.address_line1, ph.change_date, ph.old_price, ph.new_price
FROM price_history ph
JOIN vw_listing_details vld ON ph.listing_id = vld.listing_id
WHERE vld.address_line1 ILIKE '%Maple Dr%'
ORDER BY ph.change_date DESC
LIMIT 1;
```

---

**Q: Which listings have had the most price changes?**
```sql
SELECT ph.listing_id, COUNT(*) AS price_change_count
FROM price_history ph
GROUP BY ph.listing_id
ORDER BY price_change_count DESC
LIMIT 20;
```

---

**Q: What is the average price drop amount when a listing's price is reduced?**
```sql
SELECT ROUND(AVG(ph.old_price - ph.new_price)::NUMERIC, 2) AS avg_price_drop
FROM price_history ph
WHERE ph.new_price < ph.old_price;
```

---

**Q: Show all price reductions this year**
```sql
SELECT ph.listing_id, ph.change_date, ph.old_price, ph.new_price,
       (ph.old_price - ph.new_price) AS reduction_amount
FROM price_history ph
WHERE ph.new_price < ph.old_price
  AND ph.change_date >= date_trunc('year', CURRENT_DATE)
ORDER BY ph.change_date DESC
LIMIT 100;
```

---

### SHOWING QUERIES

---

**Q: What are the details of showings for a specific listing?**
```sql
SELECT sh.showing_id, sh.showing_date, sh.feedback_score, sh.feedback_notes
FROM showings sh
WHERE sh.listing_id = 101
ORDER BY sh.showing_date DESC;
```

---

**Q: Who conducted the showings for a particular listing?**
```sql
SELECT sh.showing_date, a.agent_name
FROM showings sh
LEFT JOIN agents a ON sh.agent_id = a.agent_id
WHERE sh.listing_id = 101
ORDER BY sh.showing_date DESC;
```

---

**Q: How many showings has each listing had?**
```sql
SELECT sh.listing_id, COUNT(*) AS showing_count
FROM showings sh
GROUP BY sh.listing_id
ORDER BY showing_count DESC
LIMIT 50;
```

---

**Q: What is the average showing feedback score by agent?**
```sql
SELECT a.agent_name,
       ROUND(AVG(sh.feedback_score)::NUMERIC, 1) AS avg_feedback_score,
       COUNT(sh.showing_id) AS showing_count
FROM showings sh
JOIN agents a ON sh.agent_id = a.agent_id
GROUP BY a.agent_name
ORDER BY avg_feedback_score DESC
LIMIT 20;
```

---

**Q: Which showings had feedback notes but no score?**
```sql
SELECT sh.showing_id, sh.listing_id, sh.showing_date, sh.feedback_notes
FROM showings sh
WHERE sh.feedback_notes IS NOT NULL
  AND sh.feedback_score IS NULL
ORDER BY sh.showing_date DESC
LIMIT 100;
```

---

### COMBINED / ANALYTICAL QUERIES

---

**Q: What is agent performance in terms of closed deals and sales volume?**
```sql
SELECT vap.agent_name, vap.office_name,
       vap.closed_deals, vap.total_sales_volume, vap.avg_days_on_market
FROM vw_agent_performance vap
ORDER BY vap.total_sales_volume DESC NULLS LAST
LIMIT 50;
```

---

**Q: Which agents have zero closed deals?**
```sql
SELECT vap.agent_name, vap.office_name
FROM vw_agent_performance vap
WHERE vap.closed_deals = 0
ORDER BY vap.agent_name ASC;
```

---

**Q: What is the average listing price and days on market by city?**
```sql
SELECT vms.city, vms.state, vms.listing_count,
       vms.avg_list_price, vms.avg_days_on_market
FROM vw_market_summary vms
WHERE vms.status = 'ACTIVE'
ORDER BY vms.avg_list_price DESC
LIMIT 50;
```

---

**Q: Compare average price for properties with an HOA vs without**
```sql
SELECT p.hoa_flag,
       ROUND(AVG(l.list_price)::NUMERIC, 2) AS avg_list_price,
       COUNT(*) AS listing_count
FROM listings l
JOIN properties p ON l.property_id = p.property_id
GROUP BY p.hoa_flag
ORDER BY p.hoa_flag DESC;
```

---

**Q: What is the total closed sales volume and listing count by office?**
```sql
SELECT o.office_name,
       COUNT(l.listing_id) FILTER (WHERE l.status = 'CLOSED') AS closed_listing_count,
       ROUND(SUM(l.sale_price) FILTER (WHERE l.status = 'CLOSED')::NUMERIC, 2) AS total_sales_volume
FROM offices o
LEFT JOIN listings l ON o.office_id = l.office_id
GROUP BY o.office_name
ORDER BY total_sales_volume DESC NULLS LAST;
```

---

**Q: What is the estimated commission earned by each agent on closed deals?**
```sql
SELECT a.agent_name,
       ROUND(SUM(l.sale_price * l.commission_pct / 100)::NUMERIC, 2) AS estimated_commission
FROM listings l
JOIN agents a ON l.agent_id = a.agent_id
WHERE l.status = 'CLOSED'
GROUP BY a.agent_name
ORDER BY estimated_commission DESC
LIMIT 20;
```

---

**Q: Show the showings-to-close conversion rate per listing**
```sql
SELECT l.listing_id,
       COUNT(sh.showing_id) AS showing_count,
       CASE WHEN l.status = 'CLOSED' THEN 1 ELSE 0 END AS closed
FROM listings l
LEFT JOIN showings sh ON l.listing_id = sh.listing_id
GROUP BY l.listing_id, l.status
HAVING COUNT(sh.showing_id) > 0
ORDER BY showing_count DESC
LIMIT 100;
```

---

**Q: What percentage of listings closed above, at, or below their original list price?**
```sql
SELECT
    CASE
        WHEN l.sale_price > l.original_list_price THEN 'ABOVE'
        WHEN l.sale_price = l.original_list_price THEN 'AT'
        ELSE 'BELOW'
    END AS price_outcome,
    COUNT(*) AS listing_count
FROM listings l
WHERE l.status = 'CLOSED'
GROUP BY price_outcome
ORDER BY listing_count DESC;
```

---

### VISUALIZATION-TARGETED EXAMPLE QUERIES

---

### LINE CHART QUERIES — date + single metric (2 columns)

---

**Q: Show monthly new listing count trend this year (line chart)**
```sql
SELECT date_trunc('month', l.date_listed)::date AS month,
       COUNT(*) AS new_listings
FROM listings l
WHERE l.date_listed >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show monthly closed sales volume trend this year (line chart)**
```sql
SELECT date_trunc('month', l.date_closed)::date AS month,
       SUM(l.sale_price) AS sales_volume
FROM listings l
WHERE l.status = 'CLOSED'
  AND l.date_closed >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show weekly price reduction count for the last 12 weeks (line chart)**
```sql
SELECT date_trunc('week', ph.change_date)::date AS week_start,
       COUNT(*) AS price_reductions
FROM price_history ph
WHERE ph.new_price < ph.old_price
  AND ph.change_date >= CURRENT_DATE - INTERVAL '12 weeks'
GROUP BY week_start
ORDER BY week_start ASC;
```

---

**Q: Show average days on market trend by listing month this year (line chart)**
```sql
SELECT date_trunc('month', l.date_listed)::date AS month,
       ROUND(AVG(l.days_on_market)::NUMERIC, 1) AS avg_days_on_market
FROM listings l
WHERE l.date_listed >= date_trunc('year', CURRENT_DATE)
GROUP BY month
ORDER BY month ASC;
```

---

**Q: Show daily showing count for the last 30 days (line chart)**
```sql
SELECT sh.showing_date, COUNT(*) AS showing_count
FROM showings sh
WHERE sh.showing_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sh.showing_date
ORDER BY sh.showing_date ASC;
```

---

### BAR CHART QUERIES — category + single metric, LIMIT ≤ 20 (2 columns)

---

**Q: Top 10 cities by listing count (bar chart)**
```sql
SELECT vms.city, SUM(vms.listing_count) AS listing_count
FROM vw_market_summary vms
GROUP BY vms.city
ORDER BY listing_count DESC
LIMIT 10;
```

---

**Q: Top 10 agents by closed deals (bar chart)**
```sql
SELECT vap.agent_name, vap.closed_deals
FROM vw_agent_performance vap
ORDER BY vap.closed_deals DESC
LIMIT 10;
```

---

**Q: Listing count by property type (bar chart)**
```sql
SELECT p.property_type, COUNT(l.listing_id) AS listing_count
FROM properties p
JOIN listings l ON p.property_id = l.property_id
GROUP BY p.property_type
ORDER BY listing_count DESC
LIMIT 15;
```

---

**Q: Top 10 offices by total closed sales volume (bar chart)**
```sql
SELECT o.office_name,
       SUM(l.sale_price) FILTER (WHERE l.status = 'CLOSED') AS total_sales_volume
FROM offices o
JOIN listings l ON o.office_id = l.office_id
GROUP BY o.office_name
ORDER BY total_sales_volume DESC NULLS LAST
LIMIT 10;
```

---

**Q: Average list price by property type (bar chart)**
```sql
SELECT p.property_type,
       ROUND(AVG(l.list_price)::NUMERIC, 2) AS avg_list_price
FROM properties p
JOIN listings l ON p.property_id = l.property_id
GROUP BY p.property_type
ORDER BY avg_list_price DESC
LIMIT 15;
```

---

### PIE CHART QUERIES — category + percentage/share/proportion, LIMIT ≤ 8 (2 columns)

---

**Q: Show listing status distribution (pie chart)**
```sql
SELECT l.status,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM listings l
GROUP BY l.status
ORDER BY percentage DESC
LIMIT 8;
```

---

**Q: Show property type share among all properties (pie chart)**
```sql
SELECT p.property_type,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM properties p
GROUP BY p.property_type
ORDER BY percentage DESC
LIMIT 8;
```

---

**Q: What share of properties have an HOA? (pie chart)**
```sql
SELECT p.hoa_flag,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM properties p
GROUP BY p.hoa_flag
ORDER BY percentage DESC;
```

---

**Q: Show closed sales volume share by office, top 8 (pie chart)**
```sql
SELECT o.office_name,
       ROUND((SUM(l.sale_price) FILTER (WHERE l.status = 'CLOSED'))::NUMERIC * 100.0
           / SUM((SUM(l.sale_price) FILTER (WHERE l.status = 'CLOSED'))::NUMERIC) OVER (), 1) AS sales_share
FROM offices o
JOIN listings l ON o.office_id = l.office_id
GROUP BY o.office_name
ORDER BY sales_share DESC
LIMIT 8;
```

---

**Q: Show listing source distribution (pie chart)**
```sql
SELECT COALESCE(l.source, 'Unknown') AS source,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM listings l
GROUP BY l.source
ORDER BY percentage DESC
LIMIT 8;
```

---

## END OF EXAMPLE QUERY PAIRS