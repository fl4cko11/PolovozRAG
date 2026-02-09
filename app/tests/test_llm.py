from app.core.config import Settings
from app.core.llm import get_gigachat_client
from app.core.logging import get_logger


def test_llm_response():
    settings = Settings()
    logger = get_logger(settings)

    gigachat_client = get_gigachat_client(settings, logger)

    response = gigachat_client.chat("Hello, GigaChat!")
    print(response.choices[0].message.content)

    return
