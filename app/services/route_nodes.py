from app.core.config import Settings
from app.schemas.agent_state import AgentState


class RouteNodes:
    def __init__(self, settings: Settings):
        self.settings = settings

    def grade_node(self, state: AgentState):
        for node in state.reranked_nodes:
            if node.score >= self.settings.MIN_SCORE_LEVEL:
                return {"is_enough_grade": True}
        return {"is_enough_grade": False}

    def route_after_grade(self, state: AgentState):
        if state.iteration <= self.settings.MAX_AGENT_ITTER:
            if state.is_enough_grade == True:
                return "generate"
            else:
                return "rewrite_query"
        else:
            return "generate"

    def route_after_generate(self, state: AgentState):
        if state.iteration <= self.settings.MAX_AGENT_ITTER:
            if "Не могу ответить" in state.answer:
                return "rewrite_query"
            else:
                return "postprocess"
        else:
            return "postprocess"
