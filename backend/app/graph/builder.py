"""Factory for assembling and compiling the Dependency-Inverted StateGraph."""
from typing import Any
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.core.config import get_settings
from app.core.interfaces import LLMProtocol, VectorStoreProtocol
from app.graph.edges import make_grade_and_route_edge
from app.graph.nodes.generate import make_generate_node
from app.graph.nodes.hitl_review import make_hitl_review_node
from app.graph.nodes.retrieve import make_retrieve_node
from app.graph.nodes.rewrite import make_rewrite_node
from app.graph.nodes.research_agent import make_research_agent_node
from app.graph.state import AgentState
from app.services.tool_llm import create_tool_calling_llm

def route_research(state: AgentState) -> str:
    """Route to tools if LLM requested them, else generate."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "research_tools"
    return "generate"

def build_graph(
    vector_store: VectorStoreProtocol,
    llm: LLMProtocol,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    confidence_threshold: float = 0.60,
    retrieval_top_k: int = 5,
    max_user_rules: int = 10,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(AgentState)

    settings = get_settings()
    tool_llm = create_tool_calling_llm(settings)
    research_node, research_tool_node = make_research_agent_node(tool_llm)

    # Nodes
    workflow.add_node("contextualize_query", make_rewrite_node(llm=llm))
    workflow.add_node("retrieve", make_retrieve_node(vector_store=vector_store, top_k=retrieval_top_k))
    workflow.add_node("human_review", make_hitl_review_node(max_user_rules=max_user_rules))
    workflow.add_node("generate", make_generate_node(llm=llm, max_user_rules=max_user_rules))
    workflow.add_node("research_agent", research_node)
    workflow.add_node("research_tools", research_tool_node)

    # Edges
    workflow.add_edge(START, "contextualize_query")
    workflow.add_edge("contextualize_query", "retrieve")
    
    # <-- KEY CHANGE: Route low confidence to research_agent automatically
    workflow.add_conditional_edges(
        "retrieve",
        make_grade_and_route_edge(confidence_threshold=confidence_threshold),
        {
            "generate": "generate",
            "human_review": "research_agent", 
        },
    )
    
    # Keep manual human review path intact if they explicitly approve
    workflow.add_edge("human_review", "research_agent")

    # Tool calling routing
    workflow.add_conditional_edges(
        "research_agent",
        route_research,
        {
            "research_tools": "research_tools",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("research_tools", "research_agent")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer, store=store)