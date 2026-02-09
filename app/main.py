from fastapi import FastAPI

from app.api.routers import query_router
from app.core.app import lifespan

app = FastAPI(title="PolovozAPI", lifespan=lifespan)

app.include_router(query_router)
