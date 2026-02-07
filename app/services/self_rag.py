from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.external.llm import generate_node, rewrite_query_node
from app.repositories.qdrant import retrieve_node
from app.schemas.agent_state import AgentState


def grade_node(state: AgentState):
    for node in state.reranked_nodes:
        if node.score >= settings.MIN_SCORE_LEVEL:
            return {"is_enough_grade": True}
    return {"is_enough_grade": False}


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite_query"]:
    if state.iteration <= settings.MAX_AGENT_ITTER:
        if state.is_enough_grade == True:
            return "generate"
        else:
            return "rewrite_query"
    else:
        return "generate"


def route_after_generate(state: AgentState):
    if state.iteration <= settings.MAX_AGENT_ITTER:
        if "Не могу ответить" in state.answer:
            return "rewrite_query"
        else:
            return "postprocess"
    else:
        return "postprocess"


def postprocess_node(state: AgentState) -> dict:
    """
    Заменяет все вхождения [id] в ответе на текст из reranked_nodes.
    Пример: [0] → "(источник: 'текст фрагмента под номером 0...')"
    """
    answer = state.answer or ""
    for node in state.reranked_nodes:
        placeholder = f"[{node.id}]"
        if placeholder in answer:
            citation_text = f'(источник: "{node.text.strip()}"'
            if len(citation_text) > 100:
                citation_text = citation_text[:97] + "…)"
            else:
                citation_text += ")"
            answer = answer.replace(placeholder, citation_text)

    return {"answer": answer}


builder = StateGraph(AgentState)
builder.add_node("retrieve", RunnableLambda(retrieve_node))
builder.add_node("grade", RunnableLambda(grade_node))
builder.add_node("generate", RunnableLambda(generate_node))
builder.add_node("rewrite_query", RunnableLambda(rewrite_query_node))
builder.add_node("postprocess", RunnableLambda(postprocess_node))

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
builder.add_conditional_edges(
    "generate",
    route_after_generate,
    {"rewrite_query": "rewrite_query", "postprocess": "postprocess"},
)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("postprocess", END)

graph = builder.compile()
