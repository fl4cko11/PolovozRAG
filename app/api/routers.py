from fastapi import APIRouter, HTTPException

from app.core.logging import logger
from app.schemas.agent_state import AgentState
from app.schemas.query import QueryRequest
from app.services.self_rag import graph

query_router = APIRouter(prefix="/query", tags=["query"])


@query_router.post("/ask")
async def ask_question(request: QueryRequest):
    logger.info(f"🔄 Приняли запрос /query/ask от {request.id}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Инициализация начального состояния агента
    initial_state = AgentState(
        user_query=request.user_query,
        textbook_theme=request.textbook_theme,
        highlighted_text=request.highlighted_text,
    )

    try:
        final_state = await graph.ainvoke(initial_state)
        logger.info(f"✅ Успешно обработали запрос /query/ask от {request.id}")

        return {"answer": final_state.get("answer")}

    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса /query/ask от {request.id}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}"
        ) from e
