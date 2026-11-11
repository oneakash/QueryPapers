"""Factory for assembling and compiling the Dependency-Inverted StateGraph."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.core.interfaces import LLMProtocol, VectorStoreProtocol
from app.graph.edges import make_grade_and_route_edge
from app.graph.nodes.generate import make_generate_node
from app.graph.nodes.hitl_review import make_hitl_review_node
from app.graph.nodes.retrieve import make_retrieve_node
from app.graph.nodes.rewrite import make_rewrite_node
from app.graph.state import AgentState


def build_graph(
    vector_store: VectorStoreProtocol,
    llm: LLMProtocol,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    confidence_threshold: float = 0.60,
    retrieval_top_k: int = 5,
    max_user_rules: int = 10,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble and compile the HITL RAG StateGraph with injected dependencies."""
    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(AgentState)

    workflow.add_node("contextualize_query", make_rewrite_node(llm=llm))
    workflow.add_node(
        "retrieve",
        make_retrieve_node(vector_store=vector_store, top_k=retrieval_top_k),
    )
    workflow.add_node(
        "human_review",
        make_hitl_review_node(max_user_rules=max_user_rules),
    )
    workflow.add_node(
        "generate",
        make_generate_node(llm=llm, max_user_rules=max_user_rules),
    )

    workflow.add_edge(START, "contextualize_query")
    workflow.add_edge("contextualize_query", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        make_grade_and_route_edge(confidence_threshold=confidence_threshold),
        {
            "human_review": "human_review",
            "generate": "generate",
        },
    )
    workflow.add_edge("human_review", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
