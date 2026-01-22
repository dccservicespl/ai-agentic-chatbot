from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def get_azure_llm():
    # api_key = os.environ['AZURE_OPENAI_API_KEY']
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is missing")

    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0,
    )
