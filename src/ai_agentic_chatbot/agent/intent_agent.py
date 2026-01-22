from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from ai_agentic_chatbot.agent.schema import IntentResult
from ai_agentic_chatbot.infrastructure.llm import get_azure_llm
from ai_agentic_chatbot.agent.registry import IntentType

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "custom_prompts.md"


class IntentClassifier:
    def __init__(self):
        self.llm = get_azure_llm()
        self.domain_context = PROMPT_PATH.read_text()
        self.parser = PydanticOutputParser(pydantic_object=IntentResult)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an intent classification engine.

Rules:
- Do NOT answer the user question.
- Do NOT calculate anything.
- Choose ONLY from the allowed intents.
- If unclear, return UNKNOWN.
- Output MUST follow the provided JSON schema.

Allowed intents:
{intents}

Business context:
{context}

{format_instructions}
""",
                ),
                ("human", "{question}"),
            ]
        )

    def classify(self, question: str) -> IntentResult:
        chain = (
                self.prompt
                | self.llm
                | self.parser
        )

        return chain.invoke(
            {
                "question": question,
                "context": self.domain_context,
                "intents": [intent.value for intent in IntentType],
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
