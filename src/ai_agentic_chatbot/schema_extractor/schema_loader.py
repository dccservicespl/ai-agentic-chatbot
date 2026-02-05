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
        doc_path = self.temp_dir / "schema_documentation.yaml"

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
        Uses pre-processed schema data instead of real-time extraction.
        """
        try:
            # Load processed schema documentation
            schema_doc = self.load_schema_documentation()

            table_docs = []
            for table in schema_doc.get("tables", []):
                # Create searchable text from documentation
                search_text_parts = [
                    f"Table: {table.get('table_name', '')}",
                    f"Purpose: {table.get('business_purpose', '')}",
                    f"Description: {table.get('description', '')}",
                ]

                # Add column information
                for col in table.get("columns", []):
                    col_desc = (
                        f"Column {col.get('name', '')} ({col.get('data_type', '')})"
                    )
                    if col.get("description"):
                        col_desc += f": {col.get('description')}"
                    search_text_parts.append(col_desc)

                # Add example questions
                for question in table.get("example_questions", []):
                    search_text_parts.append(f"Example: {question}")

                # Generate DDL from raw schema
                ddl = self._generate_ddl_from_doc(table)

                table_docs.append(
                    {
                        "name": table.get("table_name", ""),
                        "schema": table.get("schema_name", "public"),
                        "ddl": ddl,
                        "search_text": " ".join(search_text_parts),
                        "columns": [
                            col.get("name", "") for col in table.get("columns", [])
                        ],
                        "business_purpose": table.get("business_purpose", ""),
                        "example_questions": table.get("example_questions", []),
                    }
                )

            logger.info(f"Loaded {len(table_docs)} table documents from cache")
            return table_docs

        except Exception as e:
            logger.error(f"Failed to load cached schema data: {e}")
            # Fallback to raw schema JSON if documentation is not available
            return self._fallback_to_raw_schema()

    def _fallback_to_raw_schema(self) -> List[Dict]:
        """Fallback to raw schema JSON if documentation is not available."""
        try:
            schema_json = self.load_schema_json()

            table_docs = []
            for table in schema_json.get("tables", []):
                # Create basic search text from raw schema
                search_text_parts = [
                    f"Table: {table.get('table_name', '')}",
                    f"Schema: {table.get('schema_name', 'public')}",
                ]

                # Add column information
                for col in table.get("columns", []):
                    col_desc = (
                        f"Column {col.get('name', '')} ({col.get('data_type', '')})"
                    )
                    if not col.get("nullable", True):
                        col_desc += " NOT NULL"
                    search_text_parts.append(col_desc)

                # Add foreign key relationships
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

    def _generate_ddl_from_doc(self, table: Dict) -> str:
        """Generate DDL from processed documentation."""
        table_name = table.get("table_name", "")
        schema_name = table.get("schema_name", "public")

        lines = [f"CREATE TABLE {schema_name}.{table_name} ("]

        # Add columns
        col_lines = []
        for col in table.get("columns", []):
            col_line = f"  {col.get('name', '')} {col.get('data_type', '')}"
            if not col.get("nullable", True):
                col_line += " NOT NULL"
            if col.get("default"):
                col_line += f" DEFAULT {col.get('default')}"
            col_lines.append(col_line)

        # Add primary key
        primary_keys = table.get("primary_keys", [])
        if primary_keys:
            pk_cols = ", ".join(primary_keys)
            col_lines.append(f"  PRIMARY KEY ({pk_cols})")

        # Add foreign keys
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

        # Add columns
        col_lines = []
        for col in table.get("columns", []):
            col_line = f"  {col.get('name', '')} {col.get('data_type', '')}"
            if not col.get("nullable", True):
                col_line += " NOT NULL"
            if col.get("default"):
                col_line += f" DEFAULT {col.get('default')}"
            col_lines.append(col_line)

        # Add primary key
        primary_keys = table.get("primary_keys", [])
        if primary_keys:
            pk_cols = ", ".join(primary_keys)
            col_lines.append(f"  PRIMARY KEY ({pk_cols})")

        # Add foreign keys
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
