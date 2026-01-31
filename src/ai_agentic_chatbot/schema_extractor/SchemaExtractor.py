from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from ai_agentic_chatbot.schema_extractor import SchemaExtractionConfig
from ai_agentic_chatbot.schema_extractor.SchemaModels import DatabaseSchema, TableSchema, ColumnSchema, ForeignKeySchema


class SchemaExtractor:
    def __init__(self, engine: Engine, config: SchemaExtractionConfig):
        self.engine: Engine = engine
        self.inspector = inspect(self.engine)
        self.config = config or SchemaExtractionConfig

    def extract_database_schema(self) -> DatabaseSchema:
        database_name = self.engine.url.database
        tables: list[TableSchema] = []

        for schema_name in self._get_schemas():
            if not self._schema_allowed(schema_name):
                continue

            for table_name in self.inspector.get_table_names(schema=schema_name):
                if not self._table_allowed(table_name):
                    continue

                tables.append(
                    self._extract_table_schema(schema_name, table_name)
                )

        return DatabaseSchema(
            database_name=database_name,
            tables=tables
        )

    def _schema_allowed(self, schema_name: str) -> bool:
        if self.config.include_schemas:
            return schema_name in self.config.include_schemas
        return True

    def _table_allowed(self, table_name: str) -> bool:
        if self.config.include_tables:
            if table_name not in self.config.include_tables:
                return False

        if self.config.exclude_tables:
            if table_name in self.config.exclude_tables:
                return False

        return True

    def _get_schemas(self) -> list[str]:
        try:
            return self.inspector.get_schema_names()
        except Exception:
            return ["public"]

    def _extract_table_schema(self, schema_name: str, table_name: str) -> TableSchema:
        columns = self._extract_columns(schema_name, table_name)
        primary_keys = self._extract_primary_keys(schema_name, table_name)
        foreign_keys = self._extract_foreign_keys(schema_name, table_name)

        return TableSchema(
            schema_name=schema_name,
            table_name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys
        )

    def _extract_columns(self, schema_name: str, table_name: str) -> list[ColumnSchema]:
        column_info = self.inspector.get_columns(table_name, schema=schema_name)
        columns: list[ColumnSchema] = []

        for col in column_info:
            columns.append(
                ColumnSchema(
                    name=col["name"],
                    data_type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    default=str(col.get("default")) if col.get("default") else None
                )
            )

        return columns

    def _extract_primary_keys(self, schema_name: str, table_name: str) -> list[str]:
        pk_info = self.inspector.get_pk_constraint(table_name, schema=schema_name)
        return pk_info.get("constrained_columns", [])

    def _extract_foreign_keys(self, schema_name: str, table_name: str) -> list[ForeignKeySchema]:
        fk_info = self.inspector.get_foreign_keys(table_name, schema=schema_name)
        foreign_keys: list[ForeignKeySchema] = []

        for fk in fk_info:
            referred_table = fk.get("referred_table")
            referred_columns = fk.get("referred_columns", [])
            constrained_columns = fk.get("constrained_columns", [])

            for src_col, ref_col in zip(constrained_columns, referred_columns):
                foreign_keys.append(
                    ForeignKeySchema(
                        column=src_col,
                        referred_table=referred_table,
                        referred_column=ref_col
                    )
                )

        return foreign_keys
