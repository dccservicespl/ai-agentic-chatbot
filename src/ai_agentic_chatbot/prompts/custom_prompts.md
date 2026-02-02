## Domain Overview

This system handles sales orders, payments, and shipping data.

## Order Lifecycle

Orders can have the following statuses:

- CREATED
- PAID
- SHIPPED
- DELIVERED
- CANCELLED

## Supported Business Questions

- Monthly order counts
- Order status by order ID
- Pending orders
- Revenue summaries

## Definitions

- "Total orders" means count of orders created.
- "Sales" refers only to PAID orders.

## Constraints

- Do not calculate values.
- Do not answer user questions.
- If intent is unclear, classify as UNKNOWN.
- If the intent is a greeting, classify as GREETING.
