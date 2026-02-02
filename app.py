"""
AI Note Taker — by Alpha AI (www.alphaai.biz)

A minimal, AI-powered note-taking tool that turns text, images, or voice
recordings into structured notes with titles, summaries, key points,
and actionable items (todos, reminders, workflows).

Powered by OpenRouter or Groq — bring your own API key.
"""

import json
import base64
import uuid
import tempfile
import httpx
import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

SYSTEM_PROMPT = """\
You are an expert note-taking assistant. Given user input (which may be a \
transcription of audio, extracted text from an image, or plain text), produce \
a structured JSON object with EXACTLY this schema — no markdown fences, just \
raw JSON:

{
  "title": "short descriptive title",
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2", ...],
  "actions": {
    "todos": ["task 1", "task 2", ...],
    "reminders": ["reminder 1", ...],
    "workflows": [
      {
        "intent": "what the user wants to achieve",
        "what_to_do": "concrete steps",
        "why": "reason / motivation",
        "outcome": "expected result"
      }
    ]
  }
}

Rules:
- Always return valid JSON and nothing else.
- If a category has no items, use an empty list [].
- Be concise but informative.
- Extract actionable items aggressively — if something sounds like a task, \
reminder, or workflow, capture it.
"""

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
DEFAULT_GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

def _headers(provider: str, api_key: str) -> dict:
    h = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        h["HTTP-Referer"] = "https://huggingface.co/spaces"
        h["X-Title"] = "AI Note Taker by Alpha AI"
    return h


def transcribe_audio(api_key: str, provider: str, audio_path: str) -> str:
    """Transcribe audio using Groq Whisper or OpenRouter."""
    if provider == "groq":
        with open(audio_path, "rb") as f:
            resp = httpx.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (audio_path.split("/")[-1], f, "audio/wav")},
                data={"model": DEFAULT_GROQ_AUDIO_MODEL, "response_format": "text"},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.text.strip()
    else:
        # OpenRouter: encode audio as base64 and ask model to transcribe
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the following audio exactly. Return only the transcription text, nothing else."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        }
        resp = httpx.post(
            OPENROUTER_CHAT_URL,
            headers=_headers("openrouter", api_key),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def process_with_llm(api_key: str, provider: str, user_content, is_image: bool = False) -> dict:
    """Send content to LLM and return structured note JSON."""
    if provider == "groq":
        url = GROQ_CHAT_URL
        if is_image:
            model = DEFAULT_GROQ_VISION_MODEL
        else:
            model = DEFAULT_GROQ_MODEL
    else:
        url = OPENROUTER_CHAT_URL
        model = DEFAULT_OPENROUTER_MODEL

    if is_image and isinstance(user_content, str) and user_content.startswith("data:"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyse this image and create a structured note from its contents."},
                    {"type": "image_url", "image_url": {"url": user_content}},
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_content)},
        ]

    payload = {"model": model, "messages": messages, "temperature": 0.3}
    resp = httpx.post(url, headers=_headers(provider, api_key), json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def empty_state():
    return {"notes": [], "todos": [], "reminders": [], "workflows": []}


def add_note_to_state(state: dict, note: dict) -> dict:
    note_id = str(uuid.uuid4())[:8]
    note["id"] = note_id

    state["notes"].insert(0, note)

    for t in note.get("actions", {}).get("todos", []):
        state["todos"].insert(0, {"text": t, "note_id": note_id, "note_title": note["title"], "done": False})
    for r in note.get("actions", {}).get("reminders", []):
        state["reminders"].insert(0, {"text": r, "note_id": note_id, "note_title": note["title"]})
    for w in note.get("actions", {}).get("workflows", []):
        w["note_id"] = note_id
        w["note_title"] = note["title"]
        state["workflows"].insert(0, w)
    return state


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_notes(state: dict) -> str:
    if not state["notes"]:
        return "<div style='text-align:center;padding:40px;color:#6b7b8d;'>No notes yet. Add your first note above!</div>"
    html = ""
    for n in state["notes"]:
        kps = "".join(f"<li>{k}</li>" for k in n.get("key_points", []))
        todos = "".join(f"<li>{t}</li>" for t in n.get("actions", {}).get("todos", []))
        reminders = "".join(f"<li>{r}</li>" for r in n.get("actions", {}).get("reminders", []))
        wfs = ""
        for w in n.get("actions", {}).get("workflows", []):
            wfs += f"""<div style='background:#0d1b2a;border-radius:6px;padding:10px;margin:6px 0;'>
                <b>Intent:</b> {w.get('intent','')}<br>
                <b>What:</b> {w.get('what_to_do','')}<br>
                <b>Why:</b> {w.get('why','')}<br>
                <b>Outcome:</b> {w.get('outcome','')}
            </div>"""

        actions_section = ""
        if todos:
            actions_section += f"<div><b>📋 Todos</b><ul>{todos}</ul></div>"
        if reminders:
            actions_section += f"<div><b>🔔 Reminders</b><ul>{reminders}</ul></div>"
        if wfs:
            actions_section += f"<div><b>⚙️ Workflows</b>{wfs}</div>"

        html += f"""
        <div style='background:#1b2838;border:1px solid #2a3f5f;border-radius:10px;padding:20px;margin-bottom:16px;'>
            <h3 style='margin:0 0 8px 0;color:#7ec8e3;'>{n['title']}</h3>
            <p style='color:#c0c0c0;margin:0 0 12px 0;'>{n.get('summary','')}</p>
            <div style='margin-bottom:8px;'><b style='color:#a0c4ff;'>Key Points</b><ul style='color:#d0d0d0;margin:4px 0;'>{kps}</ul></div>
            {actions_section}
        </div>"""
    return html


def render_todos(state: dict) -> str:
    if not state["todos"]:
        return "<div style='text-align:center;padding:40px;color:#6b7b8d;'>No todos yet.</div>"
    html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;'>"
    for i, t in enumerate(state["todos"]):
        check = "✅" if t["done"] else "⬜"
        strike = "text-decoration:line-through;opacity:0.5;" if t["done"] else ""
        html += f"""
        <div style='background:#1b2838;border:1px solid #2a3f5f;border-radius:10px;padding:16px;'>
            <div style='font-size:1.3em;margin-bottom:6px;'>{check}</div>
            <div style='{strike}color:#d0d0d0;'>{t['text']}</div>
            <div style='color:#5a7a9a;font-size:0.8em;margin-top:8px;'>From: {t['note_title']}</div>
        </div>"""
    html += "</div>"
    return html


def render_reminders(state: dict) -> str:
    if not state["reminders"]:
        return "<div style='text-align:center;padding:40px;color:#6b7b8d;'>No reminders yet.</div>"
    html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;'>"
    for r in state["reminders"]:
        html += f"""
        <div style='background:#1b2838;border:1px solid #2a3f5f;border-radius:10px;padding:16px;'>
            <div style='font-size:1.3em;margin-bottom:6px;'>🔔</div>
            <div style='color:#d0d0d0;'>{r['text']}</div>
            <div style='color:#5a7a9a;font-size:0.8em;margin-top:8px;'>From: {r['note_title']}</div>
        </div>"""
    html += "</div>"
    return html


def render_workflows(state: dict) -> str:
    if not state["workflows"]:
        return "<div style='text-align:center;padding:40px;color:#6b7b8d;'>No workflows yet.</div>"
    html = ""
    for w in state["workflows"]:
        html += f"""
        <div style='background:#1b2838;border:1px solid #2a3f5f;border-radius:10px;padding:20px;margin-bottom:14px;'>
            <h4 style='color:#7ec8e3;margin:0 0 10px 0;'>⚙️ Workflow</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div style='background:#0d1b2a;border-radius:8px;padding:12px;'>
                    <div style='color:#5a9fcf;font-size:0.8em;text-transform:uppercase;'>Intent</div>
                    <div style='color:#d0d0d0;margin-top:4px;'>{w.get('intent','')}</div>
                </div>
                <div style='background:#0d1b2a;border-radius:8px;padding:12px;'>
                    <div style='color:#5a9fcf;font-size:0.8em;text-transform:uppercase;'>What to Do</div>
                    <div style='color:#d0d0d0;margin-top:4px;'>{w.get('what_to_do','')}</div>
                </div>
                <div style='background:#0d1b2a;border-radius:8px;padding:12px;'>
                    <div style='color:#5a9fcf;font-size:0.8em;text-transform:uppercase;'>Why</div>
                    <div style='color:#d0d0d0;margin-top:4px;'>{w.get('why','')}</div>
                </div>
                <div style='background:#0d1b2a;border-radius:8px;padding:12px;'>
                    <div style='color:#5a9fcf;font-size:0.8em;text-transform:uppercase;'>Outcome</div>
                    <div style='color:#d0d0d0;margin-top:4px;'>{w.get('outcome','')}</div>
                </div>
            </div>
            <div style='color:#5a7a9a;font-size:0.8em;margin-top:10px;'>From: {w.get('note_title','')}</div>
        </div>"""
    return html


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_input(text, image, audio, api_key, provider, state):
    if not api_key or not api_key.strip():
        gr.Warning("Please enter your API key first.")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), "", None, None

    api_key = api_key.strip()
    has_text = text and text.strip()
    has_image = image is not None
    has_audio = audio is not None

    if not has_text and not has_image and not has_audio:
        gr.Warning("Please provide at least one input: text, image, or audio.")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), text, image, audio

    try:
        # Determine input type and get content for the LLM
        if has_audio:
            transcript = transcribe_audio(api_key, provider, audio)
            combined = transcript
            if has_text:
                combined = f"{text.strip()}\n\n[Audio transcription]: {transcript}"
            note = process_with_llm(api_key, provider, combined)
        elif has_image:
            # Convert image to base64 data URI
            import PIL.Image as PILImage
            import io
            if isinstance(image, str):
                with open(image, "rb") as f:
                    img_bytes = f.read()
            else:
                buf = io.BytesIO()
                if hasattr(image, "save"):
                    image.save(buf, format="PNG")
                else:
                    buf.write(image)
                img_bytes = buf.getvalue()
            b64 = base64.b64encode(img_bytes).decode()
            data_uri = f"data:image/png;base64,{b64}"

            if has_text:
                # If there's also text, include it
                note = process_with_llm(api_key, provider, f"{text.strip()}\n\n[The user also attached an image — analyse it as part of the note.]", is_image=False)
                # Actually, let's use the image
                messages_content = [
                    {"type": "text", "text": f"The user provided this text along with an image: {text.strip()}\n\nAnalyse both and create a structured note."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
                note = process_with_llm(api_key, provider, data_uri, is_image=True)
            else:
                note = process_with_llm(api_key, provider, data_uri, is_image=True)
        else:
            note = process_with_llm(api_key, provider, text.strip())

        state = add_note_to_state(state, note)
        gr.Info("Note created successfully!")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), "", None, None

    except httpx.HTTPStatusError as e:
        gr.Warning(f"API error ({e.response.status_code}): {e.response.text[:200]}")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), text, image, audio
    except json.JSONDecodeError:
        gr.Warning("Failed to parse AI response. Try again.")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), text, image, audio
    except Exception as e:
        gr.Warning(f"Error: {str(e)[:200]}")
        return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state), text, image, audio


def toggle_todo(index, state):
    try:
        idx = int(index)
        if 0 <= idx < len(state["todos"]):
            state["todos"][idx]["done"] = not state["todos"][idx]["done"]
    except (ValueError, IndexError):
        pass
    return state, render_todos(state)


def clear_notes(state):
    state = empty_state()
    return state, render_notes(state), render_todos(state), render_reminders(state), render_workflows(state)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CSS = """
/* Blue-black theme overrides */
.gradio-container {
    background: #0a1628 !important;
    max-width: 1200px !important;
}
.main-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.main-header h1 {
    color: #7ec8e3 !important;
    font-size: 2em !important;
    margin-bottom: 4px !important;
}
.main-header p {
    color: #6b7b8d !important;
    font-size: 0.95em !important;
}
footer { display: none !important; }
.tab-nav button {
    color: #7ec8e3 !important;
    font-weight: 600 !important;
}
.tab-nav button.selected {
    border-color: #7ec8e3 !important;
    background: #1b2838 !important;
}
#notes-display, #todos-display, #reminders-display, #workflows-display {
    min-height: 200px;
}
.dark input, .dark textarea {
    background: #1b2838 !important;
    border-color: #2a3f5f !important;
    color: #d0d0d0 !important;
}
.dark .block {
    background: #0f1d2e !important;
    border-color: #2a3f5f !important;
}
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(css=CSS, theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0a1628",
        body_background_fill_dark="#0a1628",
        block_background_fill="#0f1d2e",
        block_background_fill_dark="#0f1d2e",
        block_border_color="#2a3f5f",
        block_border_color_dark="#2a3f5f",
        block_label_text_color="#7ec8e3",
        block_label_text_color_dark="#7ec8e3",
        block_title_text_color="#7ec8e3",
        block_title_text_color_dark="#7ec8e3",
        body_text_color="#c0c0c0",
        body_text_color_dark="#c0c0c0",
        input_background_fill="#1b2838",
        input_background_fill_dark="#1b2838",
        input_border_color="#2a3f5f",
        input_border_color_dark="#2a3f5f",
        button_primary_background_fill="#1a5276",
        button_primary_background_fill_dark="#1a5276",
        button_primary_background_fill_hover="#217dbb",
        button_primary_background_fill_hover_dark="#217dbb",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#0f1d2e",
        button_secondary_background_fill_dark="#0f1d2e",
        button_secondary_border_color="#2a3f5f",
        button_secondary_border_color_dark="#2a3f5f",
        button_secondary_text_color="#7ec8e3",
        button_secondary_text_color_dark="#7ec8e3",
    ), title="AI Note Taker — Alpha AI") as app:

        # Session state
        state = gr.State(empty_state())

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🗒️ AI Note Taker</h1>
            <p>Turn text, images & voice into structured, actionable notes — powered by AI</p>
            <p style="font-size:0.8em;color:#4a6a8a;margin-top:2px;">
                Built by <a href="https://www.alphaai.biz" target="_blank" style="color:#5a9fcf;">Alpha AI</a>
                &nbsp;·&nbsp; Bring your own API key &nbsp;·&nbsp; Your data stays in your session
            </p>
        </div>
        """)

        # API config row
        with gr.Row():
            with gr.Column(scale=1):
                provider = gr.Radio(
                    choices=["openrouter", "groq"],
                    value="openrouter",
                    label="Provider",
                    info="Choose your AI provider",
                )
            with gr.Column(scale=3):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Paste your OpenRouter or Groq API key here…",
                    info="Your key is never stored. It stays in your browser session only.",
                )

        # Tabs
        with gr.Tabs():
            # ---- Notes Tab ----
            with gr.Tab("📝 Notes", id="notes-tab"):
                gr.Markdown("*Type a note, upload an image, or record audio. The AI will structure it for you.*")
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text Input",
                            placeholder="Type or paste your note here…",
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Image Input",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Audio Input",
                            type="filepath",
                            sources=["microphone", "upload"],
                        )

                with gr.Row():
                    submit_btn = gr.Button("✨ Create Note", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")

                notes_html = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#6b7b8d;'>No notes yet. Add your first note above!</div>",
                    elem_id="notes-display",
                )

            # ---- Todos Tab ----
            with gr.Tab("📋 Todos", id="todos-tab"):
                gr.Markdown("*All your extracted to-do items, organized as cards.*")
                with gr.Row():
                    todo_index = gr.Number(label="Toggle todo # (0-indexed)", precision=0, visible=True, scale=1)
                    toggle_btn = gr.Button("Toggle Done", variant="secondary", size="sm", scale=1)
                todos_html = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#6b7b8d;'>No todos yet.</div>",
                    elem_id="todos-display",
                )

            # ---- Reminders Tab ----
            with gr.Tab("🔔 Reminders", id="reminders-tab"):
                gr.Markdown("*Reminders extracted from your notes.*")
                reminders_html = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#6b7b8d;'>No reminders yet.</div>",
                    elem_id="reminders-display",
                )

            # ---- Workflows Tab ----
            with gr.Tab("⚙️ Workflows", id="workflows-tab"):
                gr.Markdown("*Workflows extracted from your notes: intent, what to do, why, and expected outcome.*")
                workflows_html = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#6b7b8d;'>No workflows yet.</div>",
                    elem_id="workflows-display",
                )

            # ---- About Tab ----
            with gr.Tab("ℹ️ About", id="about-tab"):
                gr.Markdown("""
## About AI Note Taker

**AI Note Taker** is a smart note-taking tool that transforms unstructured input
into organized, actionable notes. Simply speak, type, or upload an image — the AI
does the rest.

### What it does
- **Accepts** text, images, or audio recordings as input
- **Generates** a structured note with a title, summary, and key points
- **Extracts** actionable items: todos, reminders, and workflows
- **Organizes** everything into dedicated tabs for easy access

### How to use
1. **Get an API key** from [OpenRouter](https://openrouter.ai/) or [Groq](https://console.groq.com/)
2. **Select your provider** and paste your key above
3. **Add input** — type text, upload an image, or record audio
4. **Click "Create Note"** and let the AI structure your note

### Privacy & Security
- Your API key is **never stored** on our servers
- Each user session is **fully isolated**
- No data persists after you close the tab

### Supported Models
| Provider | Text | Vision | Audio |
|----------|------|--------|-------|
| OpenRouter | Gemini 2.0 Flash | Gemini 2.0 Flash | Gemini 2.0 Flash |
| Groq | Llama 3.3 70B | Llama 3.2 90B Vision | Whisper Large v3 Turbo |

---

Built with ❤️ by **[Alpha AI](https://www.alphaai.biz)** — Intelligent solutions for the modern world.
                """)

        # ---- Event handlers ----
        submit_btn.click(
            fn=process_input,
            inputs=[text_input, image_input, audio_input, api_key, provider, state],
            outputs=[state, notes_html, todos_html, reminders_html, workflows_html, text_input, image_input, audio_input],
        )

        clear_btn.click(
            fn=clear_notes,
            inputs=[state],
            outputs=[state, notes_html, todos_html, reminders_html, workflows_html],
        )

        toggle_btn.click(
            fn=toggle_todo,
            inputs=[todo_index, state],
            outputs=[state, todos_html],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
