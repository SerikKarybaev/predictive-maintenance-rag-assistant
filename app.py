"""
app.py — Streamlit UI для RAG-ассистента.
Запуск: streamlit run app.py
"""

import streamlit as st
import os
from rag_logic import RAGPipeline

# ──────────────────────────────────────────────
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="🔧 Predictive Maintenance Assistant",
    page_icon="🔧",
    layout="centered",
)

st.title("🔧 Predictive Maintenance Assistant")
st.caption("Ask questions about industrial equipment maintenance, fault diagnosis, and condition monitoring")

# ──────────────────────────────────────────────
# КЭШИРОВАНИЕ ТЯЖЁЛЫХ РЕСУРСОВ
# ──────────────────────────────────────────────

@st.cache_resource
def load_pipeline(api_key: str) -> RAGPipeline:
    """
    Загружает и инициализирует RAG-пайплайн.
    @st.cache_resource гарантирует, что это выполнится ОДИН РАЗ
    (модели не будут перегружаться при каждом вопросе).
    """
    pipeline = RAGPipeline(groq_api_key=api_key)
    pipeline.initialize()
    return pipeline

# ──────────────────────────────────────────────
# ВВОД API КЛЮЧА
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Получи ключ на https://console.groq.com",
    )
    st.divider()
    st.markdown(
        """
        **Knowledge base:**
        - 📄 Condition-based maintenance
        - 📄 Explainable fault diagnosis
        - 📄 ML for Predictive Maintenance

        **Pipeline:**
        - 🧩 Parent Document Retrieval
        - 🔍 Query Expansion (LLaMA)
        - 🎯 CrossEncoder Re-ranking (top-3)
        """
    )

# ──────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ ПАЙПЛАЙНА
# ──────────────────────────────────────────────

if not api_key:
    st.warning("👈 Введи Groq API ключ в боковой панели чтобы начать.")
    st.stop()

with st.spinner("🔄 Загрузка моделей и индексация документов (первый запуск ~2 мин)..."):
    try:
        pipeline = load_pipeline(api_key)
    except Exception as e:
        st.error(f"❌ Ошибка инициализации: {e}")
        st.stop()

st.success("✅ Система готова к работе!")

# ──────────────────────────────────────────────
# ИСТОРИЯ ЧАТА
# ──────────────────────────────────────────────

# Инициализируем историю в session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ──────────────────────────────────────────────
# ВВОД ВОПРОСА
# ──────────────────────────────────────────────

if prompt := st.chat_input("Задай вопрос по RAG..."):
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            try:
                result = pipeline.ask(prompt)
                answer = result["answer"]
                sources = result["sources"]  # list of {title, page}

                st.markdown(answer)

                # Отображение источников
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                        for s in sources:
                            st.markdown(f"- **{s['title']}** — page {s['page']}")

                # Сохраняем в историю
                sources_text = "\n".join(
                    f"- *{s['title']}*, p. {s['page']}" for s in sources
                )
                full_response = f"{answer}\n\n**📎 Sources:**\n{sources_text}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Кнопка очистки истории
if st.session_state.messages:
    if st.button("🗑️ Очистить историю"):
        st.session_state.messages = []
        st.rerun()
