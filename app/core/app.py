from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings
from app.core.database import Qdrant, get_qdrant_client
from app.core.llm import get_gigachat_client
from app.core.logging import get_logger
from app.core.ml_models import get_embed_model, get_reranking_model
from app.repositories.qdrant import QdrantRetrieve
from app.services.llm_nodes import GigaChatNodes
from app.services.processing_nodes import postprocess_node
from app.services.qdrant_nodes import QdrantNodes
from app.services.route_nodes import RouteNodes
from app.services.self_rag import BuildSelfRAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logger = get_logger(settings)

    gigachat_client = get_gigachat_client(settings, logger)

    embed_model = get_embed_model(settings, logger)
    rerank_model = get_reranking_model(settings, logger)
    qdrant_client = get_qdrant_client(settings, logger)
    qdrant = Qdrant(settings, logger, qdrant_client, embed_model)
    qdrant_retriever = QdrantRetrieve(
        settings, logger, qdrant.get_qdrant_retrievers(), rerank_model
    )

    gigachat_nodes = GigaChatNodes(gigachat_client)
    qdrant_nodes = QdrantNodes(qdrant_retriever)
    route_nodes = RouteNodes(settings)

    self_rag = BuildSelfRAG(gigachat_nodes, qdrant_nodes, route_nodes, postprocess_node)

    app.state.logger = logger
    app.state.self_rag = self_rag

    yield
