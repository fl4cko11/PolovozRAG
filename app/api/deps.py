from fastapi import Request


def get_self_rag(request: Request):
    return request.app.state.self_rag


def get_logger(request: Request):
    return request.app.state.logger
