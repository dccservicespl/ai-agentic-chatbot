from typing import List, Dict
import yaml


class VectorSchemaBuilder:
    """
       Domain service responsible for:
       - Loading schema definition
       - Converting table metadata into semantic text blocks
       """

    def load_schema(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_table_text(self, schema: Dict, table: Dict) -> str:
        lines = []

        lines.append(f"Database: {schema['database_name']}")
        lines.append(f"Schema Version: {schema['version']}\n")

        lines.append(f"Table Name: {table['table_name']}\n")

        lines.append("Business Purpose:")
        lines.append(table["business_purpose"] + "\n")

        lines.append("Primary Identifier:")
        lines.append(table["primary_identifier"] + "\n")

        lines.append("Key Business Fields:")
        for field in table.get("key_fields", []):
            lines.append(f"- {field['field_name']}: {field['meaning']}")

        lines.append("\nImportant Dates:")
        for field in table.get("important_dates", []):
            lines.append(f"- {field['field_name']}: {field['meaning']}")

        if table.get("relationships"):
            lines.append("\nRelationships:")
            for rel in table["relationships"]:
                lines.append(
                    f"- Related to {rel['related_table']}: {rel['explanation']}"
                )

        if table.get("operational_notes"):
            lines.append("\nOperational Notes:")
            lines.append(table["operational_notes"])

        if table.get("example_questions"):
            lines.append("\nTypical Questions This Table Answers:")
            for q in table["example_questions"]:
                lines.append(f"- {q}")

        return "\n".join(lines)

    def build_all_tables(self, schema: Dict) -> List[Dict]:
        """
        Returns a list of:
        {
            table_name: str,
            content: str,
            metadata: dict
        }
        """
        results = []

        for table in schema["tables"]:
            content = self.build_table_text(schema, table)

            results.append(
                {
                    "table_name": table["table_name"],
                    "content": content,
                    "metadata": {
                        "database": schema["database_name"],
                        "schema_version": schema["version"],
                        "table_name": table["table_name"],
                        "object_type": "table_schema",
                    },
                }
            )

        return results
