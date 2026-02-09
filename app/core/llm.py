from logging import Logger

from gigachat import GigaChat

from app.core.config import Settings


def get_gigachat_client(settings: Settings, logger: Logger):
    try:
        logger.info("🔄 Начинаем соединение с GigaChat")
        client = GigaChat(credentials=settings.GIGACHAT_API_AUTH_KEY)
        token = client.get_token()
        token_preview = token.access_token[:10]  # Первые 10 символов
        logger.info(
            f"✅ Подключение к GigaChat успешно установлено, первые 10 символов токена: {token_preview}"
        )

        return client

    except Exception as e:
        logger.error(f"❌ Ошибка при подключении к GigaChat: {e}")
        raise
