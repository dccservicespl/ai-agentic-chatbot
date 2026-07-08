import hashlib
import json

from ai_agentic_chatbot.schema_extractor.SchemaModels import DatabaseSchema


def compute_schema_hash(schema: DatabaseSchema) -> str:
    """Deterministic SHA-256 hex digest of the structural schema only.

    Every level (tables, columns within a table, PKs, FKs) is sorted
    explicitly because introspection order is not guaranteed stable across
    runs or DB versions — without sorting, an unchanged schema could hash
    differently between two extractions.
    """
    canonical = sorted(
        (
            t.schema_name,
            t.table_name,
            t.object_type,
            sorted((c.name, c.data_type, c.nullable) for c in t.columns),
            sorted(t.primary_keys),
            sorted((fk.column, fk.referred_table, fk.referred_column) for fk in t.foreign_keys),
        )
        for t in schema.tables
    )
    payload = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()