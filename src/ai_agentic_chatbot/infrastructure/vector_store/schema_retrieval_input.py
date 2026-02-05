from pydantic import BaseModel, Field


class SchemaRetrieverInput(BaseModel):
    context: str = Field(
        ...,
        description="User question or intent used to retrieve relevant schema documents"
    )
    collection_name: str = Field(
        ...,
        description="PGVector collection name containing schema embeddings"
    )
