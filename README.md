# Білім AI

Оқу PDF материалдарына сүйеніп қазақша жауап беретін FastAPI, E5, FAISS және OpenAI үйлесімді API негізіндегі ассистент.

Толық код, орнату нұсқаулығы, тесттер мен зерттеу есебі: [ai-education-assistant](ai-education-assistant/README.md).

```bash
cd ai-education-assistant
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

`.env.example` негізінде жергілікті `.env` дайындап, `LLM_API_KEY` мәнін тек серверге беріңіз. Embedding моделін жүктегеннен кейін:

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Бұл — жергілікті зерттеу прототипі. Сыртқы модель мен API қолжетімділігі қажет; нақты LLM салыстырмалы экспериментінің орындалмаған нәтижелері толтырылмаған. Бұрынғы Codestep демосы `MVP/Index.html` ішінде сақталған.
