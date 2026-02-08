from pydantic import BaseModel


class QueryRequest(BaseModel):
    id: int
    user_query: str
    textbook_theme: str
    highlighted_text: str | None = None
