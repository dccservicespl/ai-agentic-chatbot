from pydantic import BaseModel, Field
from typing import List, Optional


class KeyField(BaseModel):
    """Represents an important business field in a table."""

    field_name: str = Field(
        description="Name of the field/column."
    )
    meaning: str = Field(
        description="Plain-English explanation of what this field represents."
    )


class ImportantDate(BaseModel):
    """Represents a date-related field and its business meaning."""

    field_name: str = Field(
        description="Name of the date field."
    )
    meaning: str = Field(
        description="Plain-English explanation of what this date represents."
    )


class RelationshipExplanation(BaseModel):
    """Represents a relationship to another table in non-technical terms."""

    related_table: str = Field(
        description="Name of the related table."
    )
    explanation: str = Field(
        description="Simple explanation of how this table relates to the other table."
    )


class TableSchemaDocumentation(BaseModel):
    """
    Human-readable, agent-safe documentation for a single database table.
    This model is designed to be produced by an LLM and consumed by other agents.
    """

    model_config = {
        "extra": "forbid"
    }

    table_name: str = Field(
        description="Name of the database table being described."
    )

    business_purpose: str = Field(
        description="Plain-English explanation of what this table represents and why it exists."
    )

    primary_identifier: str = Field(
        description="Explanation of the primary key and how it uniquely identifies records."
    )

    key_fields: List[KeyField] = Field(
        description="List of important business fields and their meanings."
    )

    important_dates: Optional[List[ImportantDate]] = Field(
        default=None,
        description=(
            "List of important date fields and their meanings. "
            "Null if the table has no major date-related fields."
        )
    )

    relationships: Optional[List[RelationshipExplanation]] = Field(
        default=None,
        description=(
            "List of relationships to other tables explained in simple terms. "
            "Null if no relationships exist."
        )
    )

    operational_notes: Optional[str] = Field(
        default=None,
        description=(
            "Explanation of operational fields such as statuses, flags, enums, limits, "
            "or workflow-related behavior. Null if none exist."
        )
    )

    example_questions: List[str] = Field(
        description=(
            "Natural language questions that this table can help answer. "
            "Written as end-user questions."
        )
    )
