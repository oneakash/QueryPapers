from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.graph.state import AgentState
from app.graph.tools.arxiv_tool import search_arxiv


def make_research_agent_node(llm: ChatOpenAI):
    """Factory for the research agent node with deterministic fallback."""
    llm_with_tools = llm.bind_tools([search_arxiv])
    tool_node = ToolNode([search_arxiv])

    async def research_agent(state: AgentState) -> dict[str, Any]:
        # Reviewer rejected context -> skip tools, let generate use fallback
        if state.get("hitl_approved") is False:
            return {"messages": []}

        query = state.get("rewritten_query") or state["original_query"]
        messages = list(state.get("messages", []))
        tool_prompt = (
            f"You are a research assistant. The user asked: '{query}'. "
            f"If the local documents are insufficient, use the search_arxiv "
            f"tool. Otherwise, respond directly without calling any tool."
        )

        try:
            response = await llm_with_tools.ainvoke(
                messages + [HumanMessage(content=tool_prompt)]
            )
        except Exception:
            # Deterministic fallback: never crash the graph.
            # route_research will see no tool_calls -> goes to generate.
            return {"messages": []}

        return {"messages": [response]}

    return research_agent, tool_node