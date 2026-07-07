### Brand/Region/Customer Master Data, Material Consumption & Sales Comparison — Context Notes

**Table alias convention:** brand_mst b, sub_brand_mst sb, region_mst r,
customer_mst c, consumption cn, sales_comparison sc

**No foreign keys / no preferred join view:** This schema declares zero FK
constraints and there is no pre-joined summary view. `brand_mst`,
`sub_brand_mst`, `region_mst`, and `customer_mst` are largely standalone
lookup/master tables. `sales_comparison` is a **denormalized fact table**
that already carries its own text copies of brand, sub-brand, region, and
customer codes/names (`brand_code`, `brand_name`, `sub_brand_code`,
`sub_brand_name`, `region_code`, `region_name`, `customer_code`,
`customer_name`, `city`, `item_code`, `item_name`) — so most "sales by
brand/region/customer" questions can be answered directly from
`sales_comparison` without joining the master tables at all.

Only join to a master table when the question specifically asks to
validate/look up canonical master data (e.g. "what is the official brand
name for code X"), and match on the code/name text columns below — never
assume an id-based FK:
- `sales_comparison.brand_code = brand_mst.brand_code` (both TEXT — safe direct join)
- `sales_comparison.sub_brand_code = sub_brand_mst.sub_brand_code` (both TEXT — safe direct join)
- `sales_comparison.customer_code = customer_mst."Customer"` (both TEXT — safe direct join; note quoted/capitalized column name)
- `sales_comparison.region_code = region_mst.region_code` — **type mismatch**: `sales_comparison.region_code` is TEXT but `region_mst.region_code` is BIGINT. Cast before joining, e.g. `sc.region_code::bigint = r.region_code`, or prefer matching on `sc.region_name = r.region_name` if codes don't cast cleanly.

`consumption` has **no** code/name columns matching any master table
(no `brand_code`, `region_code`, or `customer_code` in `consumption`) —
treat it as unrelated to `brand_mst` / `region_mst` / `customer_mst` for
joining purposes. `consumption.lifnr`/`name1` refer to a *supplier*, not a
customer. There is a `satnr` (sales order number) column in both
`consumption` and `sales_comparison`, and material-number-shaped columns
in both (`consumption.matnr_fg`/`matnr_pc` vs `sales_comparison.matnr`) —
these are *possible* informal links but are not declared relationships;
verify actual matching values before joining on them, don't assume they
correlate.

**Caution — several date and numeric fields are stored as TEXT, not a real date/numeric type:**
- `consumption.bedat` (consumption/entry date) is TEXT, not DATE/TIMESTAMP. Any date filtering or `date_trunc()` on it requires an explicit cast/parse (verify the actual stored format, e.g. `to_date(bedat, 'YYYYMMDD')`) before use — do not treat it as a native date column.
- `sales_comparison.fkdat` (billing date) and `sales_comparison.month` are also TEXT, with the same caveat.
- `sales_comparison.amount1` is TEXT while `sales_comparison.amount1_r` is DOUBLE PRECISION — prefer `amount1_r` for numeric aggregation of period-1 sales amount; only use `amount1::numeric` if the user explicitly asks for the non-reporting-unit figure.
- `consumption.menge`, `consumption.cut_qty` are TEXT despite representing quantities — cast with `::numeric` before aggregating; `consumption.erfmg`, `waste`, `wast`, `bdmng`, `tot_bom`, `tot_qty`, `diff`, `perc` are already DOUBLE PRECISION and don't need casting.
- `consumption` has **two** waste columns — `waste` and `wast` — both DOUBLE PRECISION with overlapping meaning ("amount of waste generated" vs "total waste amount recorded"). Default to `waste` unless the user's phrasing implies a running/cumulative total, in which case use `wast`.

---

**Business definitions:**

| Term                              | Exact SQL Meaning                                                        |
|------------------------------------|---------------------------------------------------------------------------|
| "brand name for a code"           | brand_mst.brand_name WHERE brand_mst.brand_code = ...                    |
| "sub-brand name for a code"       | sub_brand_mst.sub_brand_name WHERE sub_brand_mst.sub_brand_code = ...    |
| "region name for a code"          | region_mst.region_name WHERE region_mst.region_code = ...::bigint        |
| "customer name"                   | customer_mst."Name 1"                                                    |
| "customer city"                   | customer_mst."City"                                                      |
| "total consumption" / "quantity consumed" | SUM(consumption.erfmg) — already numeric; consumption.menge (TEXT) is an alternate total, cast ::numeric if user specifies it |
| "total waste"                     | SUM(consumption.waste) — see note above re: waste vs wast                |
| "cumulative waste"                | SUM(consumption.wast)                                                    |
| "cut quantity"                    | SUM(consumption.cut_qty::numeric)                                        |
| "total BOM quantity"              | SUM(consumption.tot_bom)                                                 |
| "total accounted quantity"        | SUM(consumption.tot_qty)                                                 |
| "consumption variance" / "difference" | SUM(consumption.diff)                                                |
| "consumption percentage"          | consumption.perc                                                         |
| "supplier name"                   | consumption.name1                                                        |
| "purchase order number"           | consumption.ebeln                                                        |
| "batch number"                    | consumption.charg                                                        |
| "consumption date" / "entry date" | consumption.bedat (TEXT — cast/parse required, see caution above)        |
| "sales quantity"                  | SUM(sales_comparison.qun1)                                               |
| "sales quantity (reporting unit)" | SUM(sales_comparison.qun1_r)                                             |
| "sales amount"                    | SUM(sales_comparison.amount1_r) — numeric column, preferred over TEXT amount1 |
| "planned quantity"                | SUM(sales_comparison.pqty1) — period 1; pqty2 for period 2               |
| "planned amount"                  | SUM(sales_comparison.pamt1) — period 1; pamt2 for period 2               |
| "sales by region"                 | GROUP BY sales_comparison.region_code, sales_comparison.region_name      |
| "sales by customer"               | GROUP BY sales_comparison.customer_code, sales_comparison.customer_name  |
| "sales by brand"                  | GROUP BY sales_comparison.brand_code, sales_comparison.brand_name        |
| "sales by sub-brand"              | GROUP BY sales_comparison.sub_brand_code, sales_comparison.sub_brand_name |
| "sales by item" / "sales by product" | GROUP BY sales_comparison.item_code, sales_comparison.item_name       |
| "billing date"                    | sales_comparison.fkdat (TEXT — cast/parse required)                      |
| "sales month"                     | sales_comparison.month (TEXT)                                            |
| "number of weeks"                 | sales_comparison.weeks (already INTEGER)                                 |
| "material description"            | consumption.maktx (TEXT) or sales_comparison.maktx (VARCHAR(200)) — these are separate columns in separate tables, not shared |