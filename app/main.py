from fastapi import FastAPI

from app.api.routers import query_router

app = FastAPI(title="PolovozAPI")

app.include_router(query_router)
