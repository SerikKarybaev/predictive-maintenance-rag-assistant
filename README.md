# 🔧 Predictive Maintenance RAG Assistant

RAG-чатбот для ответов на вопросы по техническому обслуживанию промышленного оборудования на основе научных документов.

## 📄 База знаний

| Файл | Описание |
|---|---|
| `1.pdf` | Condition-based maintenance in manufacturing |
| `2.pdf` | Explainable fault diagnosis |
| `3.pdf` | ML Algorithms for Predictive Maintenance |

> PDF-файлы хранятся локально в папке `data/`

## ⚙️ Архитектура

```
Вопрос → Query Expansion (LLaMA) → FAISS поиск → CrossEncoder Re-ranking → LLaMA ответ + Sources
```

- **Parent Document Retrieval** — поиск по малым чанкам, контекст из больших  
- **Query Expansion** — 3 варианта запроса для лучшего recall  
- **CrossEncoder Re-ranking** — top-3 финальных фрагмента  
- **Sources** — каждый ответ содержит документ + номер страницы

## 🛠️ Стек

`Groq (llama-3.1-8b-instant)` · `sentence-transformers (all-MiniLM-L6-v2)` · `FAISS` · `CrossEncoder` · `Streamlit` · `FastAPI` · `PyMuPDF`

## 📁 Структура

```
rag_project/
├── data/                  ← PDF файлы (положить вручную)
├── rag_logic.py           # Ядро пайплайна
├── app.py                 # Streamlit UI
├── main.py                # FastAPI сервер
├── client.py              # Python клиент
├── requirements.txt
└── README.md
```

## 🚀 Запуск

### Подготовка

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_xxxxxxx   # https://console.groq.com
```

Положи PDF в папку `data/` с именами из таблицы выше.

---

### Способ 1 — Streamlit UI

```bash
streamlit run app.py
# → http://localhost:8501
```

---

### Способ 2 — FastAPI

```bash
uvicorn main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

---

### Способ 3 — Python клиент

> FastAPI должен быть запущен

```bash
python client.py
```

---

### Способ 4 — Google Colab (через ngrok)

```python
# 1. Установка
!pip install -q groq sentence-transformers faiss-cpu PyMuPDF \
    langchain-text-splitters streamlit fastapi uvicorn pyngrok nest_asyncio

# 2. Ключи
import os
os.environ["GROQ_API_KEY"] = "gsk_xxxxxxx"
from pyngrok import conf
conf.get_default().auth_token = "xxxxxxx"  # https://ngrok.com

# 3. Загрузи PDF через Files → Upload → /content/rag_project/data/

# 4. Streamlit
import subprocess, threading, time
from pyngrok import ngrok

def run_streamlit():
    subprocess.run(["streamlit", "run", "/content/rag_project/app.py",
        "--server.port", "8501", "--server.headless", "true",
        "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"])

threading.Thread(target=run_streamlit, daemon=True).start()
time.sleep(5)
print(ngrok.connect(8501))

# 5. FastAPI (в отдельной ячейке)
import uvicorn, nest_asyncio, sys
nest_asyncio.apply()
sys.path.insert(0, "/content/rag_project")

threading.Thread(target=lambda: uvicorn.run("main:app", host="0.0.0.0", port=8000), daemon=True).start()
time.sleep(5)
api_url = ngrok.connect(8000)
print(api_url)

# 6. Клиент
import requests
response = requests.post(f"{str(api_url).rstrip('/')}/ask",
    json={"question": "What is condition-based maintenance?"})
print(response.json())
```

## 💬 Примеры вопросов

- *What is condition-based maintenance and how does it differ from preventive maintenance?*
- *What ML algorithms are most effective for predictive maintenance?*
- *How does explainable AI help in fault diagnosis?*
- *What sensors are used for equipment condition monitoring?*
