"""
rag_logic.py — RAG-модуль для Predictive Maintenance (промышленное оборудование).
Инкапсулирует всю логику: загрузка данных, индексация, поиск, генерация.
"""

import os
import fitz  # PyMuPDF
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

# ──────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────

DATA_FOLDER = "data"

# Predictive Maintenance — локальные файлы из папки data/
ARTICLES = [
    {
        "filename": "1.pdf",
        "title": "Condition-based maintenance in manufacturing industries",
    },
    {
        "filename": "2.pdf",
        "title": "Explainable fault diagnosis",
    },
    {
        "filename": "3.pdf",
        "title": "Machine Learning Algorithms for Predictive Maintenance",
    },
]

# ──────────────────────────────────────────────
# КЛАСС RAGPipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    Полный RAG-пайплайн:
      - Parent Document Retrieval
      - Query Expansion (через Groq LLaMA)
      - CrossEncoder Re-ranking
    """

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"

        # Хранилища
        self.parent_docs_store: dict[str, str] = {}
        self.child_chunks: list[str] = []
        self.child_to_parent_map: dict[int, str] = {}

        # Модели (загружаются один раз)
        print("Загрузка Embedding модели...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Загрузка CrossEncoder...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # FAISS индекс
        self.index = None

        # Сплиттеры
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        )

    # ── Загрузка и парсинг ──────────────────────

    def load_documents(self, folder_path: str = DATA_FOLDER) -> list[dict]:
        """Загружает PDF из локальной папки data/.
        Возвращает список {name, title, pages}.
        """
        title_map = {a["filename"]: a["title"] for a in ARTICLES}
        docs = []

        for article in ARTICLES:
            filepath = os.path.join(folder_path, article["filename"])
            try:
                if not os.path.exists(filepath):
                    print(f"  ⚠️  Файл не найден, пропускаю: {filepath}")
                    continue

                doc = fitz.open(filepath)
                pages = []
                for i in range(len(doc)):
                    text = doc.load_page(i).get_text()
                    if text.strip():
                        pages.append((i + 1, text))

                docs.append({
                    "name": article["filename"],
                    "title": article["title"],
                    "pages": pages,
                })
                print(f"  ✅ Загружен: {article['filename']} ({len(pages)} стр.)")

            except Exception as e:
                print(f"  ❌ Ошибка при загрузке {article['filename']}: {e}")
                continue

        if not docs:
            raise RuntimeError(
                f"Ни один PDF не загружен. Убедись, что файлы лежат в папке '{folder_path}/'"
            )

        return docs

    # ── Индексация ──────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        """Строит иерархический индекс (Parent-Child) + FAISS.
        Для каждого родительского чанка сохраняет метаданные: title + page.
        """
        doc_id_counter = 0
        self.parent_docs_store = {}
        self.parent_meta_store: dict[str, dict] = {}   # parent_id → {title, page}
        self.child_chunks = []
        self.child_to_parent_map = {}

        for doc in documents:
            for page_num, page_text in doc["pages"]:
                parent_chunks = self.parent_splitter.split_text(page_text)
                for chunk_id, parent_text in enumerate(parent_chunks):
                    parent_id = f"{doc_id_counter}_{page_num}_{chunk_id}"
                    self.parent_docs_store[parent_id] = parent_text
                    self.parent_meta_store[parent_id] = {
                        "title": doc["title"],
                        "page": page_num,
                    }
                    for child_text in self.child_splitter.split_text(parent_text):
                        idx = len(self.child_chunks)
                        self.child_chunks.append(child_text)
                        self.child_to_parent_map[idx] = parent_id

            doc_id_counter += 1

        print(f"Родительских чанков: {len(self.parent_docs_store)}")
        print(f"Дочерних чанков: {len(self.child_chunks)}")

        print("Создание эмбеддингов...")
        embeddings = self.embedding_model.encode(
            self.child_chunks, show_progress_bar=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"✅ FAISS индекс готов. Векторов: {self.index.ntotal}")

    def initialize(self) -> None:
        """Полная инициализация: загрузить локальные PDF → проиндексировать."""
        documents = self.load_documents()
        self.build_index(documents)

    # ── Поиск ───────────────────────────────────

    def _expand_query(self, query: str) -> list[str]:
        """Генерирует 3 альтернативных запроса через Groq."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in predictive maintenance and industrial equipment. "
                        "Generate 3 alternative search queries based on the original question "
                        "to improve recall in a knowledge base of technical papers on "
                        "predictive maintenance, condition monitoring, and fault diagnosis. "
                        "Output ONLY 3 questions, one per line, no numbering."
                    ),
                },
                {"role": "user", "content": f'Исходный вопрос: "{query}"'},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        expanded = [
            q.strip()
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]
        return [query] + expanded[:3]

    def _retrieve_and_rerank(
        self, query: str, k_retriever: int = 30, k_reranker: int = 3
    ) -> list[dict]:
        """Query Expansion → FAISS → CrossEncoder → родительские чанки с метаданными."""
        if self.index is None:
            raise RuntimeError("Индекс не создан. Вызовите initialize() сначала.")

        # 1. Расширение запроса
        queries = self._expand_query(query)

        # 2. Поиск по всем запросам
        query_embeddings = self.embedding_model.encode(queries)
        D, I = self.index.search(query_embeddings, k_retriever)
        all_indices = list({int(idx) for row in I for idx in row if idx >= 0})

        # 3. Re-ranking через CrossEncoder
        retrieved_chunks = [self.child_chunks[i] for i in all_indices]
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(scores, all_indices), reverse=True)

        # 4. Собираем топ-K уникальных родительских чанков с метаданными
        seen_parents: set[str] = set()
        final_chunks: list[dict] = []
        for _, child_idx in reranked:
            parent_id = self.child_to_parent_map[child_idx]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                meta = self.parent_meta_store[parent_id]
                final_chunks.append({
                    "text": self.parent_docs_store[parent_id],
                    "title": meta["title"],
                    "page": meta["page"],
                })
            if len(final_chunks) >= k_reranker:
                break

        return final_chunks

    # ── Генерация ───────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Полный RAG-цикл: вопрос → поиск → генерация ответа.
        Возвращает {"answer": str, "sources": list[{title, page}]}.
        """
        # Получаем контекст с метаданными
        chunks = self._retrieve_and_rerank(question)
        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        # Генерируем ответ
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in predictive maintenance and industrial equipment diagnostics. "
                        "Answer questions ONLY based on the provided context from technical documents. "
                        "If the information is not in the context, say so clearly. "
                        "Give structured, precise answers. Use English."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        # Формируем уникальный список источников
        seen = set()
        sources = []
        for c in chunks:
            key = (c["title"], c["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"title": c["title"], "page": c["page"]})

        return {"answer": answer, "sources": sources}
