import sys
from pathlib import Path

from gigachat import GigaChat

root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))
from app.core.config import settings

giga = GigaChat(
    credentials=settings.GIGACHAT_API_AUTH_KEY,
)

response = giga.get_token()

print(response)
