from typing import Literal

from pydantic import BaseModel, Field

# Решения маршрутизации
Decision = Literal["generate", "rewrite_query", "final_answer"]


# Сериализуемая замена NodeWithScore
class SerializableNode(BaseModel):
    id: str
    text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class AgentState(BaseModel):
    # --- Вход от фронтенда ---
    user_query: str = Field(..., description="Исходный запрос пользователя")
    highlighted_text: str | None = Field(
        None, description="Выделенный пользователем фрагмент (опционально)"
    )
    textbook_theme: str = Field(
        ..., description="Тематика учебника → имя коллекции в Qdrant"
    )

    # --- Retrieval ---
    reranked_nodes: list[SerializableNode] = Field(
        default_factory=list,
        description="Реранкированные ноды (только сериализуемые данные)",
    )

    # --- Флаги Self-RAG логики (все с default!) ---
    is_relevant: bool = Field(False, description="Есть ли релевантный контекст")
    is_enough_grade: bool = Field(
        False, description="Достаточен ли score хотя бы у одного чанка"
    )

    # --- Генерация ---
    answer: str | None = Field(None, description="Черновой ответ от LLM")

    # --- Защита от зацикливания ---
    iteration: int = Field(
        0, ge=0, description="Текущая итерация цикла retrieval → rewrite"
    )
