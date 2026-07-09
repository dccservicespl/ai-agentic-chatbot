"""Visualization node for intelligent chart type determination."""

import pandas as pd
from typing import List, Dict, Any, Literal, Optional
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


class VisualizationNode:
    """Node that analyzes query results and determines optimal visualization."""

    def __init__(self):
        pass

    def determine_visualization(
        self, state: dict, forced_type: Optional[str] = None
    ) -> dict:
        """
        Analyze query results and determine the best visualization type.
        Uses data shape, types, and content to make intelligent decisions.

        If `forced_type` is provided (used by the Refresh path), that chart
        type is attempted first; if the fresh data no longer structurally
        fits it, this falls back to the normal heuristics.
        """
        results = state.get("query_result", [])
        sql_query = state.get("generated_sql", "")
        explanation = state.get("explanation", "")

        logger.info(
            f"[Visualizer] Analyzing {len(results)} rows for visualization"
            + (f" (forced_type={forced_type!r})" if forced_type else "")
        )

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

        # Detect date columns before formatting (dd-mm-yyyy strings won't pass _is_date_column)
        date_flags = [self._is_date_column(df.iloc[:, i]) for i in range(num_cols)]
        df = self._format_date_columns(df, date_flags)

        if forced_type:
            viz_config = self._build_by_type(forced_type, df, date_flags, sql_query)
            if viz_config is not None:
                viz_config["type_reused"] = True
                logger.info(f"[Visualizer] Reused forced type '{forced_type}'")
                return {"visualization": viz_config}

            logger.info(
                f"[Visualizer] forced_type={forced_type!r} no longer fits data shape "
                "— falling back to heuristics"
            )
            viz_config = self._apply_heuristics(df, date_flags, sql_query, explanation)
            viz_config["type_reused"] = False
            return {"visualization": viz_config}

        # TODO: apply intelligent heuristics using LLMs
        viz_config = self._apply_heuristics(df, date_flags, sql_query, explanation)

        logger.info(f"[Visualizer] Selected visualization: {viz_config['type']}")

        return {"visualization": viz_config}

    logger = get_logger(__name__)

    def _apply_heuristics(
        self, df: pd.DataFrame, date_flags: list, sql_query: str, explanation: str
    ) -> dict:
        """Apply intelligent heuristics to determine visualization type."""
        sql_lower = sql_query.lower()

        sql_has_percentage = any(k in sql_lower for k in [
            "100.0", "* 100", "*100", "/ sum", "/sum", "percent"
        ])

        result = self._build_kpi(df)
        if result is not None:
            return result

        result = self._build_line_chart(df, date_flags)
        if result is not None:
            return result

        result = self._build_pie_chart(df, sql_has_percentage, require_percentage_signal=True)
        if result is not None:
            return result

        result = self._build_bar_chart(df)
        if result is not None:
            return result

        return self._build_table(df)

    def _build_by_type(
        self, viz_type: str, df: pd.DataFrame, date_flags: list, sql_query: str
    ) -> Optional[dict]:
        """Attempt to build a payload for a specific (forced) chart type.

        Returns None if the data no longer structurally fits `viz_type`, so
        the caller can fall back to `_apply_heuristics`.
        """
        sql_lower = sql_query.lower()
        sql_has_percentage = any(k in sql_lower for k in [
            "100.0", "* 100", "*100", "/ sum", "/sum", "percent"
        ])

        if viz_type == "kpi":
            return self._build_kpi(df)
        if viz_type == "line_chart":
            return self._build_line_chart(df, date_flags)
        if viz_type == "pie_chart":
            return self._build_pie_chart(df, sql_has_percentage, require_percentage_signal=False)
        if viz_type == "bar_chart":
            return self._build_bar_chart(df)
        if viz_type == "table":
            return self._build_table(df)
        return None

    def _build_kpi(self, df: pd.DataFrame) -> Optional[dict]:
        """Single Value (KPI)."""
        if df.shape != (1, 1):
            return None

        column_name = df.columns[0]
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

    def _build_line_chart(self, df: pd.DataFrame, date_flags: list) -> Optional[dict]:
        """Time Series Detection (Date + Metric) -> Line Chart."""
        if df.shape[1] != 2 or not date_flags or not date_flags[0]:
            return None

        columns = df.columns.tolist()

        return self._create_payload(
            type="line_chart",
            title=f"{self._beautify_column_name(columns[1])} over Time",
            data=df.to_dict("records"),
            summary=f"Time series showing {len(df)} data points.",
            config={
                "x_axis": columns[0],
                "y_axis": columns[1],
                "x_label": self._beautify_column_name(columns[0]),
                "y_label": self._beautify_column_name(columns[1]),
            },
        )

    def _build_pie_chart(
        self,
        df: pd.DataFrame,
        sql_has_percentage: bool,
        require_percentage_signal: bool = True,
    ) -> Optional[dict]:
        """Distribution/Percentage Data -> Pie Chart.

        `num_rows > 8` is a genuine structural fit concern (unreadable pie),
        so this cap always applies — even when reusing a forced type via
        require_percentage_signal=False.
        """
        if df.shape[1] != 2 or df.shape[0] > 8:
            return None

        columns = df.columns.tolist()
        second_col = df.iloc[:, 1]

        if require_percentage_signal:
            second_col_name = columns[1].lower()
            is_percentage_named = any(
                keyword in second_col_name
                for keyword in ["percent", "percentage", "share", "proportion"]
            )
            if not (
                is_percentage_named
                or self._is_percentage_data(second_col)
                or sql_has_percentage
            ):
                return None

        return self._create_payload(
            type="pie_chart",
            title=f"Distribution of {self._beautify_column_name(columns[0])}",
            data=df.to_dict("records"),
            summary=f"Distribution across {len(df)} categories.",
            config={
                "category": columns[0],
                "value": columns[1],
                "category_label": self._beautify_column_name(columns[0]),
                "value_label": self._beautify_column_name(columns[1]),
            },
        )

    def _build_bar_chart(self, df: pd.DataFrame) -> Optional[dict]:
        """Categorical Comparison (String + Numeric) -> Bar Chart."""
        if df.shape[1] != 2 or df.shape[0] > 20:
            return None

        columns = df.columns.tolist()
        first_col = df.iloc[:, 0]
        second_col = df.iloc[:, 1]

        if not (
            (
                pd.api.types.is_string_dtype(first_col)
                or pd.api.types.is_object_dtype(first_col)
            )
            and pd.api.types.is_numeric_dtype(second_col)
        ):
            return None

        return self._create_payload(
            type="bar_chart",
            title=f"{self._beautify_column_name(columns[1])} by {self._beautify_column_name(columns[0])}",
            data=df.to_dict("records"),
            summary=f"Comparison across {len(df)} categories.",
            config={
                "x_axis": columns[0],
                "y_axis": columns[1],
                "x_label": self._beautify_column_name(columns[0]),
                "y_label": self._beautify_column_name(columns[1]),
            },
        )

    def _build_table(self, df: pd.DataFrame) -> dict:
        """Universal fallback — never returns None.

        Detailed variant for 3+ columns up to 50 rows (matches original
        heuristic threshold), else paginated summary table.
        """
        num_rows = len(df)
        columns = df.columns.tolist()

        if df.shape[1] >= 3 and num_rows <= 50:
            return self._create_payload(
                type="table",
                title="Detailed Results",
                data=df.to_dict("records"),
                summary=f"Detailed view of {num_rows} records with {len(columns)} attributes.",
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
        if pd.api.types.is_numeric_dtype(series):
            return False
        try:
            sample_size = min(5, len(series))
            sample = series.head(sample_size)
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def _is_percentage_data(self, series) -> bool:
        """Check if numeric values represent a percentage distribution (sum ≈ 100)."""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        try:
            total = series.dropna().sum()
            return 99.0 <= float(total) <= 101.0
        except Exception:
            return False

    def _format_date_columns(self, df: pd.DataFrame, date_flags: list) -> pd.DataFrame:
        """Reformat detected date columns to dd-mm-yyyy, stripping time and timezone."""
        df = df.copy()
        for i, is_date in enumerate(date_flags):
            if is_date:
                col = df.columns[i]
                df[col] = (
                    pd.to_datetime(df[col], utc=True, errors="coerce")
                    .dt.strftime("%d-%m-%Y")
                )
        return df

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
