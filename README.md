---
title: AI Note Taker
emoji: 🗒️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Turn text, images & voice into structured, actionable notes.
---

# AI Note Taker

**Turn text, images & voice into structured, actionable notes — powered by AI.**

Built by [Alpha AI](https://www.alphaai.biz)

## What it does

AI Note Taker transforms unstructured input into organized, actionable notes. Speak, type, or upload an image — the AI structures it for you.

- **Text, Image & Audio input** — use whichever is convenient
- **Structured output** — every note gets a title, summary, key points, and extracted actions
- **Todos, Reminders & Workflows** — actionable items are automatically extracted and organized into dedicated tabs
- **Workflow extraction** — captures intent, what to do, why, and expected outcome

## How to use

1. Get a free API key from [OpenRouter](https://openrouter.ai/) or [Groq](https://console.groq.com/)
2. Select your provider and paste your key
3. Add input — type, upload an image, or record audio
4. Click **Create Note**

## Privacy

- Your API key is **never stored** — it lives only in your browser session
- Each user session is **fully isolated**
- No data persists after you close the tab

## Tech

| Provider | Text Model | Vision Model | Audio Model |
|----------|-----------|-------------|------------|
| OpenRouter | Gemini 2.0 Flash | Gemini 2.0 Flash | Gemini 2.0 Flash |
| Groq | Llama 3.3 70B | Llama 3.2 90B Vision | Whisper Large v3 Turbo |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

---

Built with ❤️ by [Alpha AI](https://www.alphaai.biz)
