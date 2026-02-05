"""Schema loader utility to read pre-processed schema data."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ai_agentic_chatbot.logging_config import get_logger
import os

logger = get_logger(__name__)


class SchemaLoader:
    """Utility class to load pre-processed schema data."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.temp_dir = self.base_dir / "temp"

    def load_schema_json(self) -> Dict:
        """Load the raw schema JSON data."""
        schema_path = self.temp_dir / "db_schema.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema JSON not found at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_schema_documentation(self) -> Dict:
        """Load the processed schema documentation YAML."""
        doc_path = Path(os.environ["SCHEMA_PATH"])

        if not doc_path.exists():
            raise FileNotFoundError(f"Schema documentation not found at {doc_path}")

        with open(doc_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_schema_summary(self) -> Dict:
        """Load the schema summary for router hints."""
        try:
            summary_path = os.environ.get("SCHEMA_SUMMARY_PATH")
            if not summary_path:
                raise ValueError("SCHEMA_SUMMARY_PATH environment variable not set")

            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load schema summary: {e}")
            return {}

    def get_table_docs_for_search(self) -> List[Dict]:
        """
        Get table documents formatted for semantic search.
        Uses pre-processed schema data with rich business context.
        """
        try:
            schema_doc = self.load_schema_documentation()

            table_docs = []
            for table in schema_doc.get("tables", []):
                search_text_parts = [
                    f"Table: {table.get('table_name', '')}",
                    f"Business Purpose: {table.get('business_purpose', '')}",
                    f"Primary Identifier: {table.get('primary_identifier', '')}",
                ]

                for field in table.get("key_fields", []):
                    field_desc = f"Field {field.get('field_name', '')}: {field.get('meaning', '')}"
                    search_text_parts.append(field_desc)

                for date_field in table.get("important_dates", []):
                    date_desc = f"Date {date_field.get('field_name', '')}: {date_field.get('meaning', '')}"
                    search_text_parts.append(date_desc)

                relationships = table.get("relationships")
                if relationships:
                    for rel in relationships:
                        rel_desc = f"Related to {rel.get('related_table', '')}: {rel.get('explanation', '')}"
                        search_text_parts.append(rel_desc)

                if table.get("operational_notes"):
                    search_text_parts.append(f"Notes: {table.get('operational_notes')}")

                for question in table.get("example_questions", []):
                    search_text_parts.append(f"Example Query: {question}")

                ddl = self._generate_ddl_from_your_format(table)

                columns = []
                for field in table.get("key_fields", []):
                    columns.append(field.get("field_name", ""))
                for date_field in table.get("important_dates", []):
                    columns.append(date_field.get("field_name", ""))

                table_docs.append(
                    {
                        "name": table.get("table_name", ""),
                        "schema": "public",
                        "ddl": ddl,
                        "search_text": " ".join(search_text_parts),
                        "columns": columns,
                        "business_purpose": table.get("business_purpose", ""),
                        "example_questions": table.get("example_questions", []),
                        "key_fields": table.get("key_fields", []),
                        "relationships": table.get("relationships", []),
                        "operational_notes": table.get("operational_notes", ""),
                    }
                )

            logger.info(
                f"Loaded {len(table_docs)} table documents from preprocessed schema"
            )
            return table_docs

        except Exception as e:
            logger.error(f"Failed to load preprocessed schema data: {e}")
            return self._fallback_to_raw_schema()

    def _fallback_to_raw_schema(self) -> List[Dict]:
        """Fallback to raw schema JSON if documentation is not available."""
        try:
            schema_json = self.load_schema_json()

            table_docs = []
            for table in schema_json.get("tables", []):
                search_text_parts = [
                    f"Table: {table.get('table_name', '')}",
                    f"Schema: {table.get('schema_name', 'public')}",
                ]

                for col in table.get("columns", []):
                    col_desc = (
                        f"Column {col.get('name', '')} ({col.get('data_type', '')})"
                    )
                    if not col.get("nullable", True):
                        col_desc += " NOT NULL"
                    search_text_parts.append(col_desc)

                for fk in table.get("foreign_keys", []):
                    search_text_parts.append(
                        f"Foreign key {fk.get('column', '')} references "
                        f"{fk.get('referred_table', '')}.{fk.get('referred_column', '')}"
                    )

                ddl = self._generate_ddl_from_raw(table)

                table_docs.append(
                    {
                        "name": table.get("table_name", ""),
                        "schema": table.get("schema_name", "public"),
                        "ddl": ddl,
                        "search_text": " ".join(search_text_parts),
                        "columns": [
                            col.get("name", "") for col in table.get("columns", [])
                        ],
                        "business_purpose": "",
                        "example_questions": [],
                    }
                )

            logger.info(
                f"Loaded {len(table_docs)} table documents from raw schema (fallback)"
            )
            return table_docs

        except Exception as e:
            logger.error(f"Failed to load raw schema data: {e}")
            return []

    def _generate_ddl_from_your_format(self, table: Dict) -> str:
        """Generate DDL from your preprocessed schema format."""
        table_name = table.get("table_name", "")

        lines = [f"CREATE TABLE {table_name} ("]

        col_lines = []

        for field in table.get("key_fields", []):
            field_name = field.get("field_name", "")
            data_type = self._infer_data_type(field_name, field.get("meaning", ""))
            col_line = f"  {field_name} {data_type}"
            col_lines.append(col_line)

        for date_field in table.get("important_dates", []):
            field_name = date_field.get("field_name", "")
            col_line = f"  {field_name} TIMESTAMP"
            col_lines.append(col_line)

        primary_identifier = table.get("primary_identifier", "")
        if "id" in primary_identifier.lower():
            col_lines.append("  PRIMARY KEY (id)")

        lines.append(",\n".join(col_lines))
        lines.append(");")

        return "\n".join(lines)

    def _infer_data_type(self, field_name: str, meaning: str) -> str:
        """Infer SQL data type from field name and meaning."""
        field_lower = field_name.lower()
        meaning_lower = meaning.lower()

        if field_lower == "id" or field_lower.endswith("_id"):
            return "INTEGER"

        if "code" in field_lower:
            return "VARCHAR(50)"

        if "status" in field_lower or "is_" in field_lower:
            return "VARCHAR(20)"

        if any(
            word in field_lower
            for word in ["cost", "price", "total", "limit", "amount"]
        ):
            return "DECIMAL(10,2)"

        if "quantity" in field_lower or "on_hand" in field_lower:
            return "INTEGER"

        if "name" in field_lower:
            return "VARCHAR(255)"

        if (
            "date" in field_lower
            or "created_at" in field_lower
            or "updated_at" in field_lower
        ):
            return "TIMESTAMP"

        return "VARCHAR(255)"

    def _generate_ddl_from_doc(self, table: Dict) -> str:
        """Generate DDL from processed documentation (legacy method)."""
        table_name = table.get("table_name", "")
        schema_name = table.get("schema_name", "public")

        lines = [f"CREATE TABLE {schema_name}.{table_name} ("]

        col_lines = []
        for col in table.get("columns", []):
            col_line = f"  {col.get('name', '')} {col.get('data_type', '')}"
            if not col.get("nullable", True):
                col_line += " NOT NULL"
            if col.get("default"):
                col_line += f" DEFAULT {col.get('default')}"
            col_lines.append(col_line)

        primary_keys = table.get("primary_keys", [])
        if primary_keys:
            pk_cols = ", ".join(primary_keys)
            col_lines.append(f"  PRIMARY KEY ({pk_cols})")

        for fk in table.get("foreign_keys", []):
            fk_line = (
                f"  FOREIGN KEY ({fk.get('column', '')}) "
                f"REFERENCES {fk.get('referred_table', '')}({fk.get('referred_column', '')})"
            )
            col_lines.append(fk_line)

        lines.append(",\n".join(col_lines))
        lines.append(");")

        return "\n".join(lines)

    def _generate_ddl_from_raw(self, table: Dict) -> str:
        """Generate DDL from raw schema JSON."""
        table_name = table.get("table_name", "")
        schema_name = table.get("schema_name", "public")

        lines = [f"CREATE TABLE {schema_name}.{table_name} ("]

        col_lines = []
        for col in table.get("columns", []):
            col_line = f"  {col.get('name', '')} {col.get('data_type', '')}"
            if not col.get("nullable", True):
                col_line += " NOT NULL"
            if col.get("default"):
                col_line += f" DEFAULT {col.get('default')}"
            col_lines.append(col_line)

        primary_keys = table.get("primary_keys", [])
        if primary_keys:
            pk_cols = ", ".join(primary_keys)
            col_lines.append(f"  PRIMARY KEY ({pk_cols})")

        for fk in table.get("foreign_keys", []):
            fk_line = (
                f"  FOREIGN KEY ({fk.get('column', '')}) "
                f"REFERENCES {fk.get('referred_table', '')}({fk.get('referred_column', '')})"
            )
            col_lines.append(fk_line)

        lines.append(",\n".join(col_lines))
        lines.append(");")

        return "\n".join(lines)


# TODO: This global instance relating to the process... multi tenancy should not be having hat
_schema_loader: Optional[SchemaLoader] = None


def get_schema_loader() -> SchemaLoader:
    """Get the global schema loader instance."""
    global _schema_loader
    if _schema_loader is None:
        _schema_loader = SchemaLoader()
    return _schema_loader
