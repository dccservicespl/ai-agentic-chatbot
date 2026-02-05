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
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_TO_TEXT_PROMPT_PATH = BASE_DIR / "prompts" / "schema_to_text_prompts.md"
USER_SCHEMA_TO_TEXT_PROMPT_PATH = BASE_DIR / "prompts" / "user_schema_to_text_prompt.md"
DB_SCHEMA_JSON_PATH = BASE_DIR / "temp" / "db_schema.json"
YAML_OUT_PATH = BASE_DIR / "temp"


def transform_schema_to_text() -> None:
    llm = get_llm()
    db_schema_json = json.loads(Path(DB_SCHEMA_JSON_PATH).read_text(encoding="utf-8"))
    system_prompt = load_file_content(SCHEMA_TO_TEXT_PROMPT_PATH)
    validated_tables: List[TableSchemaDocumentation] = []
    schema_summary = {}

    for table in db_schema_json["tables"]:
        table_json = json.dumps(table, indent=2)

        user_prompt = load_file_content(USER_SCHEMA_TO_TEXT_PROMPT_PATH).format(
            table_json=table_json
        )
        structured_llm = llm.with_structured_output(
            TableSchemaDocumentation, strict=True
        )

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

    #
    # write_text_file(
    #     directory=YAML_OUT_PATH,
    #     filename="schema_documentation.yaml",
    #     content=yaml_payload,
    # )

    yaml_text = yaml.safe_dump(
        yaml_payload,
        sort_keys=False,
        allow_unicode=True,
    )

    dir_path = Path(YAML_OUT_PATH)
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / "schema_documentation.yaml"
    file_path.write_text(yaml_text, encoding="utf-8")


logger = get_logger(__name__)


def generate_schema_summary() -> None:
    with open(os.environ["SCHEMA_PATH"], "r") as f:
        schema = yaml.safe_load(f)

    logger.info(schema)
    schema_summary = {
        "database_name": schema.get("database_name"),
        "version": "v1",
        "tables": [
            {
                "table": table.get("table_name"),
                "bussiness_purpose": table.get("business_purpose"),
                "example_questions": table.get("example_questions"),
            }
            for table in schema.get("tables")
        ],
    }
    with open(os.environ["SCHEMA_SUMMARY_PATH"], "w+") as f:
        json.dump(schema_summary, f)


def load_schema_summary() -> str:
    with open(os.environ["SCHEMA_SUMMARY_PATH"], "r") as f:
        return json.load(f)


if __name__ == "__main__":
    load_dotenv()
    generate_schema_summary()
