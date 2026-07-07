import json
from pathlib import Path
from typing import List

from ai_agentic_chatbot.logging_config import get_logger
from dotenv import load_dotenv
import yaml
from pydantic import ValidationError

from ai_agentic_chatbot.infrastructure.llm import get_llm
from ai_agentic_chatbot.schema_extractor.table_schema_documentation import (
    TableSchemaDocumentation,
)
from ai_agentic_chatbot.utils.prompt_loader import load_file_content

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_TO_TEXT_PROMPT_PATH = BASE_DIR / "prompts" / "schema_to_text_prompts.md"
USER_SCHEMA_TO_TEXT_PROMPT_PATH = BASE_DIR / "prompts" / "user_schema_to_text_prompt.md"


def transform_schema_to_text(schema_dir: Path) -> None:
    schema_dir = Path(schema_dir)
    db_schema_json_path = schema_dir / "db_schema.json"
    llm = get_llm()
    db_schema_json = json.loads(db_schema_json_path.read_text(encoding="utf-8"))
    system_prompt = load_file_content(SCHEMA_TO_TEXT_PROMPT_PATH)
    validated_tables: List[TableSchemaDocumentation] = []
    schema_summary = {}

    for table in db_schema_json["tables"]:
        table_json = json.dumps(table, indent=2)

        user_prompt = load_file_content(USER_SCHEMA_TO_TEXT_PROMPT_PATH).format(
            table_json=table_json
        )
        structured_llm = llm.with_structured_output(TableSchemaDocumentation)

        # decision = structured_llm.invoke([system_prompt, user_prompt, db_schema_json])
        decision = structured_llm.invoke([system_prompt, user_prompt])
        try:
            table_doc = TableSchemaDocumentation.model_validate(decision)
            validated_tables.append(table_doc)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON for table '{table.get('table_name')}'"
            ) from e

        except ValidationError as e:
            raise ValueError(
                f"LLM output failed schema validation for table '{table.get('table_name')}'"
            ) from e

    # Convert Pydantic models → dicts
    yaml_payload = {
        "database_name": db_schema_json.get("database_name"),
        "version": "v1",
        "tables": [table.model_dump() for table in validated_tables],
    }

    yaml_text = yaml.safe_dump(
        yaml_payload,
        sort_keys=False,
        allow_unicode=True,
    )

    schema_dir.mkdir(parents=True, exist_ok=True)

    file_path = schema_dir / "schema_documentation.yaml"
    file_path.write_text(yaml_text, encoding="utf-8")


logger = get_logger(__name__)


def generate_schema_summary(schema_dir: Path) -> None:
    schema_dir = Path(schema_dir)
    schema_doc_path = schema_dir / "schema_documentation.yaml"
    schema_summary_path = schema_dir / "schema_summary.json"

    with open(schema_doc_path, "r") as f:
        schema = yaml.safe_load(f)

    schema_summary = {
        "database_name": schema.get("database_name"),
        "version": "v1",
        "tables": [
            {
                "table": table.get("table_name"),
                "bussiness_purpose": table.get("business_purpose"),
                "example_questions": table.get("example_questions"),
            }
            for table in schema.get("tables", [])
        ],
    }
    schema_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_summary_path, "w") as f:
        json.dump(schema_summary, f)


def load_schema_summary(schema_dir: Path) -> dict:
    schema_summary_path = Path(schema_dir) / "schema_summary.json"
    with open(schema_summary_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    load_dotenv()
    generate_schema_summary(Path(__file__).resolve().parent.parent.parent.parent / "temp")
