# 📚 AI-помощник для учебников

Интерактивная система для изучения учебников с поддержкой RAG и анализом выделенных фрагментов текста.

![Стек](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Qdrant%20%7C%20LlamaIndex%20%7C%20LangGraph%20%7C%20GigaChatAPI%20%7C%20LangSmith-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)

## 🌟 Функционал

### Бэкенд
- **Векторное хранилище**: Qdrant с предзагруженными учебниками по разным тематикам
- **Self-RAG агент**:
  - Преобразование запроса в эмбеддинг
  - Поиск релевантных чанков в векторной БД
  - Реранкинг результатов
  - Формирование промпта и генерация ответа через LLM
  - Агентская рефлексия и самокоррекция

## 👀 Структура проекта

```text
.
├── app
│   ├── __init__.py
│   ├── api                 # FastAPI сервер
│   │   ├── deps.py
│   │   └── routers.py
│   ├── app.log
│   ├── core
│   │   ├── app.py          # lifespan приложения
│   │   ├── config.py       # конфигурация проекта
│   │   ├── database.py     # подключение к БД
│   │   ├── llm.py          # подключение у LLM API
│   │   ├── logging.py
│   │   ├── ml_models
│   │   │   ├── embeddings
│   │   │   └── reranking
│   │   └── ml_models.py    # подключение локальных моделей к LlamaIndex
│   ├── main.py
│   ├── repositories
│   │   └── qdrant.py       # работа с qdrant
│   ├── schemas
│   │   ├── agent_state.py  # схема состояния агента для LangGraph
│   │   └── query.py        # схема тела запросов к API
│   ├── services
│   │   ├── llm_nodes.py
│   │   ├── processing_nodes.py
│   │   ├── qdrant_nodes.py
│   │   ├── route_nodes.py
│   │   └── self_rag.py     # stateful граф SelfRAG агента на LangGraph
│   ├── tests
│   │   ├── test_agent.py
│   │   ├── test_connections.py
│   │   ├── test_llm.py
│   │   └── test_retrieve.py
│   └── utils
│       └── validators.py
├── README.md
├── requirements.txt
└── scripts
    ├── datasets
    │   ├── main_datasets
    │   │   └── polovoz.pdf
    │   └── test_datasets
    │       └── polovoz_test.pdf
    ├── download_models.py
    └── ingestion_qdrant.py
```