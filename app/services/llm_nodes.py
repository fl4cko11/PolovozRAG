from gigachat import GigaChat

from app.schemas.agent_state import AgentState


class GigaChatNodes:
    def __init__(self, gigachat_client: GigaChat):
        self.gigachat_client = gigachat_client

    def generate_node(self, state: AgentState):
        generate_prompt = f"""Ты — эксперт в области {state.textbook_theme}". Твоя задача — дать точный, краткий и цитируемый ответ на вопрос пользователя, относящийся к выделенному тексту, ОБЯЗАТЕЛЬНО используя информацию из предоставленных фрагментов (каждый с id) и в конце ответа обязательно в ответе сделай ссылку на фрагмент в формате [id]. Если информация недостаточна — скажи "Не могу ответить на основе учебника".

    Вопрос пользователя: "{state.user_query}"
    Выделенный текст: {state.highlighted_text}

    Фрагменты (id → текст):
    {"\n".join([f"[{i}]: {n.text}" for i, n in enumerate(state.reranked_nodes)])}

    ИНСТРУКЦИЯ:
    1. Ответ должен быть на русском.
    2. Не выдумывай факты. Если нет подтверждения — напиши В ТОЧНОСТИ ЭТУ ФРАЗУ: «Не могу ответить на основе учебника».
    """
        response = self.gigachat_client.chat(generate_prompt)

        return {"answer": response.choices[0].message.content}

    def rewrite_query_node(self, state: AgentState):
        rewrite_query_prompt = f"""Ты — эксперт в области {state.textbook_theme}". Пользователь задал вопрос: "{state.user_query} по этой теме: {state.highlighted_text}".

    Эта информация НЕ содержит достаточной информации для ответа:
    {'\n'.join([f"[{i}]: {n.text[:150]}..." for i, n in enumerate(state.reranked_nodes[:3])])}

    Задача: перепиши запрос так, чтобы он:
    1. Был конкретным, технически точным и использовал термины из учебника.
    2. Включал возможные синонимы/формулировки из извлечённых фрагментов.

    Верни ТОЛЬКО новый запрос (без пояснений). Примеры:
    - Было: "Что такое предел последовательности чисел?" → Стало: "Формальное определение предела последовательности чисел"
    """
        response = self.gigachat_client.chat(rewrite_query_prompt)

        return {
            "user_query": response.choices[0].message.content,
            "iteration": state.iteration + 1,
        }
