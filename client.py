"""
client.py — Простой клиент к RAG FastAPI сервису.
Запуск: python client.py

Требования: сервер должен быть запущен (uvicorn main:app --port 8000)
"""

import requests
import json

# ──────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────

BASE_URL = "http://localhost:8000"


# ──────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────

def check_health() -> bool:
    """Проверяет, что сервер запущен и индекс готов."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        data = response.json()

        if data["status"] == "ok":
            print(f"✅ Сервер готов | Векторов в индексе: {data['vectors_count']}")
            return True
        else:
            print(f"⚠️  Сервер запущен, но индекс не готов: {data}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Не удалось подключиться к {BASE_URL}")
        print("   Запусти сервер: uvicorn main:app --reload --port 8000")
        return False


def ask(question: str) -> dict | None:
    """
    Отправляет вопрос на /ask и возвращает ответ.
    """
    payload = {"question": question}

    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=60,  # RAG может занять время
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Ошибка сервера {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("❌ Timeout — сервер не ответил за 60 секунд")
        return None
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return None


def print_answer(result: dict) -> None:
    """Красиво выводит ответ."""
    print("\n" + "="*60)
    print(f"❓ Вопрос: {result['question']}")
    print("="*60)
    print(f"💡 Ответ:\n{result['answer']}")
    print("-"*60)
    print(f"📎 Использовано фрагментов: {result['sources_count']}")
    print("="*60 + "\n")


# ──────────────────────────────────────────────
# ГЛАВНАЯ ЛОГИКА
# ──────────────────────────────────────────────

def main():
    print("\n🤖 RAG Assistant Client")
    print("="*60)

    # 1. Проверяем сервер
    if not check_health():
        return

    # 2. Тестовые вопросы (можно заменить своими)
    test_questions = [
        "Что такое Parent Document Retrieval и зачем он нужен?",
        "Как работает Self-RAG и чем отличается от обычного RAG?",
        "Какие метрики используются для оценки RAG-систем?",
    ]

    print(f"\n📋 Отправляю {len(test_questions)} тестовых вопроса...\n")

    for question in test_questions:
        print(f"⏳ Обрабатываю: '{question[:50]}...'")
        result = ask(question)
        if result:
            print_answer(result)

    # 3. Интерактивный режим
    print("\n💬 Интерактивный режим (введи 'exit' для выхода)")
    while True:
        user_input = input("\nТвой вопрос: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 До свидания!")
            break
        if not user_input:
            continue

        result = ask(user_input)
        if result:
            print_answer(result)


if __name__ == "__main__":
    main()
