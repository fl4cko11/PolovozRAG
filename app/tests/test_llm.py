from gigachat import GigaChat

from app.core.config import settings

with GigaChat(credentials=settings.GIGACHAT_API_AUTH_KEY) as client:
    response = client.chat("Hello, GigaChat!")
    print(response.choices[0].message.content)
