"""
main.py — FastAPI сервер для RAG-ассистента.
Запуск: uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_logic import RAGPipeline

# ──────────────────────────────────────────────
# PYDANTIC МОДЕЛИ (валидация запросов/ответов)
# ──────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Тело запроса к /ask."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Вопрос к RAG-системе",
        example="Что такое Query Expansion в RAG?",
    )


class Source(BaseModel):
    """Один источник: документ + страница."""
    title: str = Field(..., description="Название документа")
    page: int = Field(..., description="Номер страницы")


class AnswerResponse(BaseModel):
    """Тело ответа от /ask."""
    question: str = Field(..., description="Исходный вопрос")
    answer: str = Field(..., description="Сгенерированный ответ")
    sources: list[Source] = Field(..., description="Источники: документ + страница")


class HealthResponse(BaseModel):
    """Ответ health-check эндпоинта."""
    status: str
    index_ready: bool
    vectors_count: int


# ──────────────────────────────────────────────
# LIFESPAN: инициализация при старте сервера
# ──────────────────────────────────────────────

# Глобальный пайплайн (загружается один раз при старте)
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализируем RAG при старте, очищаем при остановке."""
    global rag_pipeline

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Переменная окружения GROQ_API_KEY не задана! "
            "Задай её перед запуском: export GROQ_API_KEY=your_key"
        )

    print("🚀 Инициализация RAG-пайплайна...")
    rag_pipeline = RAGPipeline(groq_api_key=api_key)
    rag_pipeline.initialize()
    print("✅ RAG-пайплайн готов. Сервер запущен.")

    yield  # Сервер работает

    # Cleanup при остановке (опционально)
    rag_pipeline = None
    print("🛑 Сервер остановлен.")


# ──────────────────────────────────────────────
# ПРИЛОЖЕНИЕ
# ──────────────────────────────────────────────

app = FastAPI(
    title="Промышленный RAG-ассистент",
    description="API для вопросов по научным статьям о RAG-системах",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Эндпоинты ───────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Корневой эндпоинт — информация об API."""
    return {
        "service": "RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/ask", "/health"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Проверка состояния сервиса."""
    if rag_pipeline is None or rag_pipeline.index is None:
        return HealthResponse(status="not_ready", index_ready=False, vectors_count=0)

    return HealthResponse(
        status="ok",
        index_ready=True,
        vectors_count=rag_pipeline.index.ntotal,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
    """
    Основной эндпоинт: принимает вопрос, возвращает ответ RAG-системы.

    - **question**: вопрос пользователя (3–1000 символов)
    - **answer**: ответ, сгенерированный LLaMA на основе найденных фрагментов
    - **sources_count**: количество использованных родительских чанков
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG-пайплайн не инициализирован")

    try:
        result = rag_pipeline.ask(request.question)
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
