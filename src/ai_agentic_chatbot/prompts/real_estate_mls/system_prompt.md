### Real Estate MLS-Style System — Context Notes

**Table alias convention:** offices o, agents a, properties p, listings l,
price_history ph, showings sh, vw_listing_details vld, vw_market_summary
vms, vw_agent_performance vap

**Preferred views — use these instead of manually joining when possible:**
- `vw_listing_details` — the general-purpose flattened join of
  `listings JOIN properties JOIN agents JOIN offices`. Use this for any
  question that needs listing + property + agent + office fields together
  (address, price, agent name, office name, property characteristics),
  instead of chaining four manual joins. Note: it carries `market_area`
  from `offices` but **not** `region` — if a question needs `region`,
  join `offices` directly even when otherwise using this view.
- `vw_market_summary` — pre-aggregated by `city, state, status`:
  `listing_count`, `avg_list_price`, `avg_sale_price`,
  `avg_days_on_market`, `avg_price_per_sqft`. Prefer this over a manual
  GROUP BY for "market conditions in <city>" style questions.
- `vw_agent_performance` — pre-aggregated per agent:
  `agent_id, agent_name, office_name, closed_deals, total_sales_volume,
  avg_days_on_market`. All four numeric columns are computed only over
  `listings.status = 'CLOSED'` rows (via `FILTER`) — an agent with only
  active/expired listings shows `0`/`NULL` here, not their full pipeline.
  It joins to `offices` via `agents.office_id`, **not** `listings.office_id`
  — `office_name` here is the agent's current office, which may differ
  from the office recorded on any specific listing.

There is **no office-level performance view** — "office performance"
questions must be computed ad hoc from `listings` (or `vw_agent_performance`
grouped by `office_name`), there is nothing pre-built for it.

**Cleanly typed schema — no casting needed:** every date-like column
(`hire_date`, `date_listed`, `date_closed`, `date_expired`, `change_date`,
`showing_date`) is a native `DATE`, `created_at`/`updated_at` are native
`TIMESTAMP`, all price/measurement columns are native `NUMERIC`, and
`is_active` / `hoa_flag` are native `BOOLEAN`. No `::numeric` / `::date`
casts are required anywhere in this schema.

**Foreign keys (all clean INTEGER = INTEGER, no type mismatches):**
- `agents.office_id = offices.office_id` — nullable, use LEFT JOIN (an agent may have no office)
- `listings.property_id = properties.property_id`
- `listings.agent_id = agents.agent_id`
- `listings.office_id = offices.office_id`
- `price_history.listing_id = listings.listing_id`
- `showings.listing_id = listings.listing_id`
- `showings.agent_id = agents.agent_id` — nullable, use LEFT JOIN (a showing may have no recorded agent)

`listings.office_id` and `agents.office_id` are two independent paths to
`offices` — a listing's office is not guaranteed to match its agent's
current office, so don't treat them as redundant when both appear in a query.

**Listing status values are only partially confirmed.** `listings.status`
(VARCHAR(20), NOT NULL) defaults to `'ACTIVE'` and has no CHECK constraint
or enum type. `vw_agent_performance`'s definition hard-codes
`FILTER (WHERE l.status = 'CLOSED')`, which **confirms `'CLOSED'`** as a
real value (used for closed/sold deals) — this is the one additional
value backed by actual SQL, not just narrative. Beyond `'ACTIVE'` and
`'CLOSED'`, the schema documentation is inconsistent (it mentions "active,
sold, expired" in one place and "active, pending, sold" in another,
neither matching the confirmed `'CLOSED'` value) — treat any other status
string (`EXPIRED`, `PENDING`, `SOLD`, `WITHDRAWN`) as unconfirmed. Prefer
`listings.date_expired IS NOT NULL` as the reliable signal for expired
listings rather than guessing a status string for it.

**Price fields — which one to use:**
- `listings.list_price` — current/latest asking price (use for "listing price" / "asking price" now)
- `listings.original_list_price` — the initial asking price when first listed
- `listings.sale_price` — final transacted price; NULL unless `status = 'CLOSED'`
- `listings.price_per_sqft` — pre-computed and stored; prefer it directly over recalculating from `properties.square_footage`
- `listings.days_on_market` — stored/maintained column; prefer it over computing `date_closed - date_listed` or `CURRENT_DATE - date_listed` yourself; only fall back to computing if it's NULL
- `price_history.old_price` / `price_history.new_price` — a per-change audit trail; use only when the question is about price change history/trend/count, not current price (for "original list price" use `listings.original_list_price` directly, not `MIN(price_history.change_date)`)

**Derived metrics with no stored column** — compute these via expression, don't look for a column:
- commission amount: `listings.sale_price * listings.commission_pct / 100`
- property age: `EXTRACT(YEAR FROM CURRENT_DATE) - properties.year_built`
- showing-to-close conversion: join `showings.listing_id = listings.listing_id` and compare listings with at least one showing against those with `status = 'CLOSED'` — no dedicated column/view for this

---

**Business definitions:**

| Term                              | Exact SQL Meaning                                                        |
|------------------------------------|---------------------------------------------------------------------------|
| "active listings"                 | listings.status = 'ACTIVE'                                               |
| "closed listings" / "sold listings" | listings.status = 'CLOSED'                                             |
| "expired listings"                | listings.date_expired IS NOT NULL                                        |
| "listing price" / "asking price"  | listings.list_price                                                      |
| "original list price"             | listings.original_list_price                                             |
| "sale price" / "final sale price" | listings.sale_price                                                      |
| "price per square foot"           | listings.price_per_sqft                                                  |
| "days on market"                  | listings.days_on_market                                                  |
| "average days on market"          | AVG(listings.days_on_market), or vw_market_summary.avg_days_on_market / vw_agent_performance.avg_days_on_market when already aggregated |
| "commission rate"                 | listings.commission_pct                                                  |
| "commission amount"               | listings.sale_price * listings.commission_pct / 100                     |
| "price reduction" / "price drop"  | price_history.new_price < price_history.old_price                       |
| "price increase"                  | price_history.new_price > price_history.old_price                       |
| "price change amount"             | price_history.new_price - price_history.old_price                       |
| "number of price changes"         | COUNT(price_history.price_change_id) GROUP BY price_history.listing_id  |
| "agent performance"               | vw_agent_performance (closed_deals, total_sales_volume, avg_days_on_market) |
| "closed deals" (per agent)        | vw_agent_performance.closed_deals                                        |
| "agent's total sales volume"      | vw_agent_performance.total_sales_volume, or SUM(listings.sale_price) WHERE agent_id = ... AND status = 'CLOSED' |
| "listings per agent"              | COUNT(listings.listing_id) GROUP BY listings.agent_id                    |
| "active agents"                   | agents.is_active = TRUE                                                  |
| "agent hire date" / "tenure"      | agents.hire_date                                                         |
| "agent license number"            | agents.license_number                                                    |
| "office performance"              | no dedicated view — aggregate listings or vw_agent_performance GROUP BY office_name/office_id |
| "listings per office"             | COUNT(listings.listing_id) GROUP BY listings.office_id                   |
| "office region"                   | offices.region (not available via vw_listing_details — join offices directly) |
| "office market area"              | offices.market_area                                                      |
| "brokerage"                       | offices.brokerage_name                                                   |
| "market summary" / "market conditions in a city" | vw_market_summary (grouped by city, state, status)          |
| "average listing price by city"   | vw_market_summary.avg_list_price                                         |
| "number of showings for a listing" | COUNT(showings.showing_id) GROUP BY showings.listing_id                |
| "average showing feedback"        | AVG(showings.feedback_score)                                             |
| "showing feedback notes"          | showings.feedback_notes                                                  |
| "property size" / "square footage" | properties.square_footage                                              |
| "lot size"                        | properties.lot_size_sqft                                                 |
| "bedrooms" / "bathrooms"          | properties.bedrooms, properties.bathrooms                                |
| "year built"                      | properties.year_built                                                    |
| "property age"                    | EXTRACT(YEAR FROM CURRENT_DATE) - properties.year_built                  |
| "HOA fee"                         | properties.hoa_fee WHERE properties.hoa_flag = TRUE                      |
| "property type"                   | properties.property_type                                                 |
| "school district"                 | properties.school_district                                               |
| "zoning"                          | properties.zoning                                                        |
| "property location" (city/county/state/zip) | properties.city, properties.county, properties.state, properties.zip_code |
| "listing source"                  | listings.source                                                          |
| "parking spaces"                  | properties.parking_spaces                                                |
