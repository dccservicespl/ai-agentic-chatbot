import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_azure_openai_embedding():
    """Create Azure OpenAI embedding client."""
    model = os.getenv("EMBEDDING_MODEL_NAME")
    api_key = (os.getenv("EMBEDDING_API_KEY"))
    endpoint = os.getenv("EMBEDDING_ENDPOINT")
    api_version = os.getenv("EMBEDDING_API_VERSION")
    timeout = os.getenv("EMBEDDING_TIMEOUT")
    max_retries = os.getenv("EMBEDDING_MAX_RETRIES")

    embeddings = AzureOpenAIEmbeddings(
        model=model,
        azure_endpoint=endpoint,
        api_key=api_key,
        openai_api_version=api_version,
    )

    return embeddings
