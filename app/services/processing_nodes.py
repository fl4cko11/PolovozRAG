from app.schemas.agent_state import AgentState


def postprocess_node(state: AgentState):
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
