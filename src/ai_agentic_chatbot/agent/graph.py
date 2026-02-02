from langchain_core.messages import SystemMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

from ai_agentic_chatbot.agent.router import RouterNode, RouterDecision
from ai_agentic_chatbot.agent.state import AgentState
from ai_agentic_chatbot.infrastructure.llm import get_llm
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType

fast_llm = get_llm(provider=LLMProvider.AZURE_OPENAI, model=ModelType.FAST)


def router(state: AgentState) -> dict:
    return RouterNode(state).classify()


def greeting_node(state: AgentState) -> dict:
    prompt = SystemMessage(
        "You are a helpful chat assistant. User has greeted you. Greet them warmly and ask how can you help them."
    )
    response = fast_llm.invoke([prompt, *state["messages"]])
    return {"messages": [response.content]}


def fallback_node(state: AgentState) -> dict:
    return {
        "messages": ["I am sorry I could not understand that. Can you please rephrase?"]
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("fallback_node", fallback_node)

    workflow.add_edge(START, "router_node")

    def routing_policy(state: AgentState) -> str:
        return state["next_step"]

    workflow.add_conditional_edges(
        "router_node",
        routing_policy,
        {
            "greeting": "greeting_node",
            "idiotic": "fallback_node",
            "__end__": END,
        },
    )

    workflow.add_edge("greeting_node", END)
    workflow.add_edge("fallback_node", END)

    return workflow.compile()
