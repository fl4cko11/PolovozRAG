from collections.abc import Callable

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.schemas.agent_state import AgentState
from app.services.llm_nodes import GigaChatNodes
from app.services.qdrant_nodes import QdrantNodes
from app.services.route_nodes import RouteNodes


def BuildSelfRAG(
    llm_nodes: GigaChatNodes,
    qdrant_nodes: QdrantNodes,
    route_nodes: RouteNodes,
    postprocess_node: Callable[[AgentState], AgentState],
):

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", RunnableLambda(qdrant_nodes.retrieve_node))
    builder.add_node("grade", RunnableLambda(route_nodes.grade_node))
    builder.add_node("generate", RunnableLambda(llm_nodes.generate_node))
    builder.add_node("rewrite_query", RunnableLambda(llm_nodes.rewrite_query_node))
    builder.add_node("postprocess", RunnableLambda(postprocess_node))

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_nodes.route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    builder.add_conditional_edges(
        "generate",
        route_nodes.route_after_generate,
        {"rewrite_query": "rewrite_query", "postprocess": "postprocess"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("postprocess", END)

    graph = builder.compile()

    return graph
