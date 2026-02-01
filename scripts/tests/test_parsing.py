import sys
from pathlib import Path

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file import PDFReader

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from app.utils.validators import ensure_path_exists


def test_print_first_3_chunks_5_percent():
    try:

        pdf_path = (
            Path(__file__).parent.parent
            / "datasets"
            / "test_datasets"
            / "polovoz_test.pdf"
        )
        loc_pdf = ensure_path_exists(pdf_path)

        print(f"🔄 Загружаем PDF: {loc_pdf.name}")
        reader = PDFReader()
        documents = reader.load_data(file=loc_pdf)
        print(f"✅ Загружено {len(documents)} страниц\n")

        print("🔄 Разбиваем на чанки...")
        parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])
        nodes = parser.get_nodes_from_documents(documents)
        print(f"✅ Создано {len(nodes)} чанков\n")

        # Берём первые 3 чанка
        for i, node in enumerate(nodes[:3]):
            text = node.text.strip()
            length = len(text)
            sample_size = max(1, int(length * 0.05))  # 5% от длины чанка
            sample_text = text[:sample_size]

            print(f"--- Чанк {i+1} (длина: {length} символов) ---")
            print(sample_text)
            print("\n" + "-" * 50 + "\n")

    except Exception as e:
        print(f"❌ Ошибка парсинга: {e}")


if __name__ == "__main__":
    test_print_first_3_chunks_5_percent()
