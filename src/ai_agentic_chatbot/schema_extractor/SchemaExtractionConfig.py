from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SchemaExtractionConfig:
    include_schemas: Optional[List[str]] = None
    include_tables: Optional[List[str]] = None
    exclude_tables: Optional[List[str]] = None
