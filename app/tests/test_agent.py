from app.services.self_rag import graph

graph.invoke({"user_query": "что такое комплексное число", "textbook_theme": "math"})
