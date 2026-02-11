"""Visualization node for intelligent chart type determination."""

import pandas as pd
from typing import List, Dict, Any, Literal, Optional
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


class VisualizationNode:
    """Node that analyzes query results and determines optimal visualization."""

    def __init__(self):
        pass

    def determine_visualization(self, state: dict) -> dict:
        """
        Analyze query results and determine the best visualization type.
        Uses data shape, types, and content to make intelligent decisions.
        """
        results = state.get("query_result", [])
        sql_query = state.get("generated_sql", "")
        explanation = state.get("explanation", "")

        logger.info(f"[Visualizer] Analyzing {len(results)} rows for visualization")

        if not results:
            return {
                "visualization": {
                    "type": "text",
                    "title": "No Results",
                    "content": "No data found for this query.",
                    "data": [],
                    "columns": [],
                    "config": {},
                    "summary": "Query returned no results.",
                }
            }

        df = pd.DataFrame(results)
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()

        logger.info(
            f"[Visualizer] Data shape: {num_rows}x{num_cols}, columns: {columns}"
        )

        # TODO: apply intelligent heuristics using LLMs
        viz_config = self._apply_heuristics(df, sql_query, explanation)

        logger.info(f"[Visualizer] Selected visualization: {viz_config['type']}")

        return {"visualization": viz_config}

    logger = get_logger(__name__)

    def _apply_heuristics(
        self, df: pd.DataFrame, sql_query: str, explanation: str
    ) -> dict:
        """Apply intelligent heuristics to determine visualization type."""
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        sql_lower = sql_query.lower()

        logger.info("df", df.head())
        # Single Value (KPI)
        if num_rows == 1 and num_cols == 1:
            column_name = columns[0]
            value = df.iloc[0, 0]

            formatted_value = self._format_kpi_value(value, column_name)

            return self._create_payload(
                type="kpi",
                title=self._beautify_column_name(column_name),
                data=df.to_dict("records"),
                summary=f"The {self._beautify_column_name(column_name).lower()} is {formatted_value}.",
                config={
                    "value": formatted_value,
                    "metric": column_name,
                    "format": self._detect_value_format(value, column_name),
                },
            )

        # 2. Time Series Detection (Date + Metric) -> Line Chart
        if num_cols == 2:
            first_col_data = df.iloc[:, 0]
            if self._is_date_column(first_col_data):
                return self._create_payload(
                    type="line_chart",
                    title=f"{self._beautify_column_name(columns[1])} over Time",
                    data=df.to_dict("records"),
                    summary=f"Time series showing {num_rows} data points.",
                    config={
                        "x_axis": columns[0],
                        "y_axis": columns[1],
                        "x_label": self._beautify_column_name(columns[0]),
                        "y_label": self._beautify_column_name(columns[1]),
                    },
                )

        # Categorical Comparison (String + Numeric) -> Bar Chart
        if num_cols == 2 and num_rows <= 20:
            first_col = df.iloc[:, 0]
            second_col = df.iloc[:, 1]

            if (
                pd.api.types.is_string_dtype(first_col)
                or pd.api.types.is_object_dtype(first_col)
            ) and pd.api.types.is_numeric_dtype(second_col):

                return self._create_payload(
                    type="bar_chart",
                    title=f"{self._beautify_column_name(columns[1])} by {self._beautify_column_name(columns[0])}",
                    data=df.to_dict("records"),
                    summary=f"Comparison across {num_rows} categories.",
                    config={
                        "x_axis": columns[0],
                        "y_axis": columns[1],
                        "x_label": self._beautify_column_name(columns[0]),
                        "y_label": self._beautify_column_name(columns[1]),
                    },
                )

        # Distribution/Percentage Data -> Pie Chart
        if num_cols == 2 and num_rows <= 8:
            second_col_name = columns[1].lower()
            if any(
                keyword in second_col_name
                for keyword in ["percent", "percentage", "share", "proportion"]
            ):
                return self._create_payload(
                    type="pie_chart",
                    title=f"Distribution of {self._beautify_column_name(columns[0])}",
                    data=df.to_dict("records"),
                    summary=f"Distribution across {num_rows} categories.",
                    config={
                        "category": columns[0],
                        "value": columns[1],
                        "category_label": self._beautify_column_name(columns[0]),
                        "value_label": self._beautify_column_name(columns[1]),
                    },
                )

        # Multiple Metrics (3+ columns, few rows) -> Table with highlights
        if num_cols >= 3 and num_rows <= 50:
            return self._create_payload(
                type="table",
                title="Detailed Results",
                data=df.to_dict("records"),
                summary=f"Detailed view of {num_rows} records with {num_cols} attributes.",
                config={
                    "columns": columns,
                    "highlight_numeric": True,
                    "sortable": True,
                },
            )

        return self._create_payload(
            type="table",
            title="Query Results",
            data=df.head(100).to_dict("records"),
            summary=(
                f"Showing first 100 of {num_rows} records."
                if num_rows > 100
                else f"All {num_rows} records displayed."
            ),
            config={
                "columns": columns,
                "total_rows": num_rows,
                "paginated": num_rows > 100,
                "sortable": True,
            },
        )

    def _is_date_column(self, series) -> bool:
        """Check if a pandas series contains datetime-like data."""
        try:
            sample_size = min(5, len(series))
            sample = series.head(sample_size)
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def _format_kpi_value(self, value: Any, column_name: str) -> str:
        """Format KPI values based on column name context."""
        if not isinstance(value, (int, float)):
            return str(value)

        column_lower = column_name.lower()

        if any(
            keyword in column_lower
            for keyword in [
                "sales",
                "revenue",
                "amount",
                "price",
                "cost",
                "total",
                "value",
            ]
        ):
            return f"${value:,.2f}"

        if any(keyword in column_lower for keyword in ["percent", "rate", "ratio"]):
            if 0 <= value <= 1:
                return f"{value:.1%}"
            else:
                return f"{value:.1f}%"

        if any(
            keyword in column_lower
            for keyword in ["count", "number", "qty", "quantity"]
        ):
            return f"{int(value):,}"

        if value >= 1000:
            return f"{value:,.2f}"
        else:
            return f"{value:.2f}"

    def _detect_value_format(self, value: Any, column_name: str) -> str:
        """Detect the format type for frontend styling."""
        if not isinstance(value, (int, float)):
            return "text"

        column_lower = column_name.lower()

        if any(
            keyword in column_lower
            for keyword in [
                "sales",
                "revenue",
                "amount",
                "price",
                "cost",
                "total",
                "value",
            ]
        ):
            return "currency"
        elif any(keyword in column_lower for keyword in ["percent", "rate", "ratio"]):
            return "percentage"
        elif any(
            keyword in column_lower
            for keyword in ["count", "number", "qty", "quantity"]
        ):
            return "integer"
        else:
            return "decimal"

    def _beautify_column_name(self, column_name: str) -> str:
        """Convert column names to human-readable titles."""
        return column_name.replace("_", " ").replace("-", " ").title()

    def _create_payload(
        self,
        type: str,
        title: str,
        data: List[Dict],
        summary: str = "",
        config: Optional[Dict] = None,
    ) -> dict:
        """Create standardized visualization payload."""
        return {
            "type": type,
            "title": title,
            "data": data,
            "columns": list(data[0].keys()) if data else [],
            "config": config or {},
            "summary": summary,
            "row_count": len(data),
        }


def visualizer_node(state: dict) -> dict:
    """Node function for the LangGraph workflow."""
    visualizer = VisualizationNode()
    return visualizer.determine_visualization(state)
