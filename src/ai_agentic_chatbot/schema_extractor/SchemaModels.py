from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ColumnSchema:
    name: str
    data_type: str
    nullable: bool
    default: Optional[str]


@dataclass
class ForeignKeySchema:
    column: str
    referred_table: str
    referred_column: str


@dataclass
class TableSchema:
    schema_name: str
    table_name: str
    columns: List[ColumnSchema]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeySchema]


@dataclass
class DatabaseSchema:
    database_name: str
    tables: List[TableSchema]
