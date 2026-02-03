from dataclasses import asdict
import json
import tempfile
from pathlib import Path
import os
from typing import Union, Any, Literal

import yaml

from ai_agentic_chatbot.schema_extractor.SchemaModels import DatabaseSchema


def save_schema_temp_file(schema: DatabaseSchema) -> Path:
    schema_path = get_schema_file_path()
    schema_dict = asdict(schema)

    with tempfile.NamedTemporaryFile(
            mode="w",
            dir=schema_path.parent,
            delete=False,
            encoding="utf-8"
    ) as tmp_file:
        json.dump(schema_dict, tmp_file, indent=2)
        temp_path = Path(tmp_file.name)

    os.replace(temp_path, schema_path)  # atomic on all major OS
    return schema_path


def serialize_schema(schema: DatabaseSchema) -> dict:
    return asdict(schema)


def get_schema_file_path() -> Path:
    temp_dir = Path.cwd().resolve() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / "db_schema.json"


def get_temp_dir() -> Path:
    temp_dir = get_project_root() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_project_root() -> Path:
    return Path.cwd().resolve()


def write_text_file(
        *,
        directory: Union[str, Path],
        filename: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
) -> Path:
    """
    Write text content to a file with explicit directory and filename.

    Args:
        directory: Target directory.
        filename: File name (with extension).
        content: Text content to write.
        encoding: File encoding.
        create_dirs: Create directory if missing.

    Returns:
        Path to the written file.
    """
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a simple file name, not a path.")

    # if not content.strip():
    #     raise ValueError("Refusing to write empty content.")

    dir_path = Path(directory)
    if create_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / filename
    file_path.write_text(content, encoding=encoding)

    return file_path


def serialize_data(
        data: Any,
        format: Literal["json", "yaml", "text"],
) -> str:
    """
    Serialize structured data into string format.

    Args:
        data: Structured data (dict, list, or str).
        format: Target serialization format.

    Returns:
        Serialized string.
    """
    if format == "text":
        if not isinstance(data, str):
            raise TypeError("Text format requires data to be str.")
        return data

    if format == "json":
        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )

    if format == "yaml":
        return yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
        )

    raise ValueError(f"Unsupported serialization format: {format}")
