from dataclasses import asdict
import json
import tempfile
from pathlib import Path

from ai_agentic_chatbot.schema_extractor.SchemaModels import DatabaseSchema


def save_schema_temp_file(schema: DatabaseSchema) -> Path:
    schema_dict = serialize_schema(schema)
    temp_dir = get_temp_dir()

    with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_db_schema.json",
            prefix="schema_",
            dir=temp_dir,
            delete=False,
            encoding="utf-8"
    ) as tmp_file:
        json.dump(schema_dict, tmp_file, indent=2)
        return Path(tmp_file.name)


def serialize_schema(schema: DatabaseSchema) -> dict:
    return asdict(schema)


def get_temp_dir() -> Path:
    temp_dir = get_project_root() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_project_root() -> Path:
    return Path.cwd().resolve()
