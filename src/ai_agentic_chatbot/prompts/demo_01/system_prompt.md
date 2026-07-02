### Sales Order Management & Inventory System — Context Notes

**Preferred view:** Use `v_sales_summary` for any query that touches
more than one table. It pre-joins customer, orders, sales, product,
and inventory and is optimized for analytical queries.

**Table alias convention:** customer c, orders o, sales s, product p,
inventory i, v_sales_summary v

**Date column priority:** use `order_date` unless `delivery_date` is
explicitly mentioned.

---

**Order lifecycle:**

```
PENDING → ORDERED → PROCESSING → PROCESSED → POSTED → DELIVERED → COMPLETED
                                                                 ↘ VOID (any stage)
```

| Status     | Meaning                                          |
|------------|---------------------------------------------------|
| PENDING    | Order received, not yet confirmed                |
| ORDERED    | Order confirmed by system                        |
| PROCESSING | Order being picked or prepared                   |
| PROCESSED  | Order preparation complete                       |
| POSTED     | Order dispatched or invoiced                     |
| DELIVERED  | Order physically delivered to customer           |
| COMPLETED  | Order fully closed and settled                   |
| VOID       | Order cancelled or invalidated at any stage      |

---

**Business definitions:**

| Term                    | Exact SQL Meaning                                              |
|-------------------------|------------------------------------------------------------------|
| "revenue"               | SUM(sales.quantity * sales.unit_price)                         |
| "total sales"           | SUM(sales.quantity * sales.unit_price)                         |
| "sales value"           | SUM(sales.quantity * sales.unit_price)                         |
| "sales volume"          | SUM(sales.quantity) — total units sold                         |
| "order count"           | COUNT(DISTINCT orders.order_no)                                |
| "line item count"       | COUNT(sales.id)                                                |
| "average order value"   | AVG(orders.grand_total)                                        |
| "gross profit"          | SUM((sell_for - base_cost) * quantity)                         |
| "stock value"           | SUM(inventory.on_hand * inventory.base_cost)                   |
| "active customers"      | customer.activation_status = 'ACTIVE'                          |
| "inactive customers"    | customer.activation_status = 'INACTIVE'                        |
| "pending orders"        | orders.order_status = 'PENDING'                                |
| "completed orders"      | orders.order_status = 'COMPLETED'                              |
| "void orders"           | orders.order_status = 'VOID'                                   |
| "delivered orders"      | orders.order_status = 'DELIVERED'                              |
| "orders in progress"    | orders.order_status IN ('PROCESSING', 'PROCESSED', 'POSTED')   |
| "processed sales lines" | sales.sales_status = 'PROCESSED'                                |
| "available stock"       | inventory.available_quantity                                   |
| "low stock"             | inventory.available_quantity < 50                              |
| "out of stock"          | inventory.available_quantity = 0                                |
| "expected stock"        | inventory.expected_quantity                                    |
| "new arrivals"          | product.new_arrival = TRUE                                     |
| "unavailable products"  | product.status = 'OUT'                                          |
| "available products"    | product.status = 'CALL'                                         |
| "credit hold customers" | customer.credit_status = 'HOLD'                                 |
| "today"                 | = CURRENT_DATE                                                  |
| "this week"             | >= date_trunc('week', CURRENT_DATE)                              |
| "this month"            | >= date_trunc('month', CURRENT_DATE)                             |
| "last month"            | BETWEEN date_trunc('month', CURRENT_DATE - INTERVAL '1 month') AND date_trunc('month', CURRENT_DATE) - INTERVAL '1 day' |
| "this year"             | >= date_trunc('year', CURRENT_DATE)                              |
| "last year"             | BETWEEN date_trunc('year', CURRENT_DATE - INTERVAL '1 year') AND date_trunc('year', CURRENT_DATE) - INTERVAL '1 day' |
| "last 7 days"           | >= CURRENT_DATE - INTERVAL '7 days'                              |
| "last 30 days"          | >= CURRENT_DATE - INTERVAL '30 days'                             |
| "last 90 days"          | >= CURRENT_DATE - INTERVAL '90 days'                             |