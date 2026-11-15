from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolNode
from langchain_openai import ChatOpenAI
from app.graph.state import AgentState
from app.graph.tools.arxiv_tool import search_arxiv

def make_research_agent_node(llm: ChatOpenAI):
    """Factory for the research agent node (mirrors your make_generate_node pattern)."""
    llm_with_tools = llm.bind_tools([search_arxiv])  # <-- SHOWCASE: bind_tools
    tool_node = ToolNode([search_arxiv])              # <-- SHOWCASE: ToolNode
    
    async def research_agent(state: AgentState) -> dict[str, Any]:
        """Decide whether to call ArXiv tool or pass through to generate."""
        # If HITL rejected, skip tool calling and let generate handle fallback
        if state.get("hitl_approved") is False:
            return {"messages": []}
        
        messages = list(state.get("messages", []))
        query = state.get("rewritten_query") or state["original_query"]
        
        # Prepare prompt for tool-calling decision
        tool_prompt = (
            f"You are a research assistant. The user asked: '{query}'. "
            f"If the local documents are insufficient, use the search_arxiv tool. "
            f"Otherwise, respond directly."
        )
        
        if not messages or messages[-1].type != "human":
            messages.append(HumanMessage(content=tool_prompt))
        
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    return research_agent, tool_node