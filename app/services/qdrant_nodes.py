from app.repositories.qdrant import QdrantRetrieve
from app.schemas.agent_state import AgentState, SerializableNode


class QdrantNodes:
    def __init__(self, qdrant_retriever: QdrantRetrieve):
        self.qdrant_retriever = qdrant_retriever

    def retrieve_node(self, state: AgentState):
        reranked_nodes = self.qdrant_retriever.retrieve_nodes_with_rerank(
            state.user_query, state.textbook_theme
        )

        # Преобразуем NodeWithScore → SerializableNode
        serializable_nodes = [
            SerializableNode(
                id=n.node_id,
                text=n.get_content(),
                score=getattr(n, "score", None),
                metadata=n.metadata or {},
            )
            for n in reranked_nodes
        ]

        return {"reranked_nodes": serializable_nodes}
