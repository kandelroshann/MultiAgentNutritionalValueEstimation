import os
import json
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from main import (
    process_image_bytes,
    create_chat_session_from_result,
)


def init_env() -> None:
    try:
        load_dotenv()
    except Exception:
        pass


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your environment or a .env file.")
        st.stop()


def ensure_session_state() -> None:
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role: "user"|"assistant", content: str}
    if "did_upload" not in st.session_state:
        st.session_state.did_upload = False
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "uploaded_mime_type" not in st.session_state:
        st.session_state.uploaded_mime_type = None


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --accent: #2563eb; /* neutral primary */
          --accent-2: #0ea5e9;
          --bg-start: #ffffff;
          --bg-end: #ffffff;
          --card-bg: #ffffff;
        }

        /* Global background + text color */
        html, body {
          background: #ffffff !important;
          color: #0f172a !important;  /* 👈 dark text */
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .block-container,
        [data-testid="stMarkdownContainer"],
        .stMarkdown {
          background: #ffffff !important;
          color: #0f172a !important;  /* 👈 dark text */
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
          color: #111827 !important;
        }

        /* Simple white background */
        body::before { content: none !important; }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
          background: #ffffff !important; box-shadow: none !important; border: 0 !important;
        }
        .stApp { background: #ffffff !important; }
        .block-container { padding-top: 1.2rem; }

        /* Top app header styling and toolbar background */
        [data-testid="stHeader"] { background: #ffffff !important; }
        [data-testid="stHeader"] [role="button"] {
          background: #f1f5f9 !important; color: #334155 !important;
          border-radius: 999px !important; padding: 6px 10px !important; margin-left: 6px !important; border: 1px solid #e2e8f0 !important;
        }
        [data-testid="stHeader"] [role="button"] svg,
        [data-testid="stHeader"] [role="button"] path { fill: #334155 !important; stroke: #334155 !important; }

        /* Hide sidebar entirely */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        .stSidebar { background: #ffffff; }

        .stButton>button { border-radius: 10px; border: 1px solid var(--accent); }

        /* Chat input styling - simple neutral */
        [data-testid="stChatInput"] {
          background: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          border-radius: 999px !important;
          padding: 6px 8px !important;
          box-shadow: none !important;
        }
        [data-testid="stChatInput"] input {
          border-radius: 999px !important; border: none !important;
          background: transparent !important; outline: none !important;
          color: #0f172a !important;  /* 👈 dark text in input */
        }
        [data-testid="stChatInput"] input::placeholder { color: #64748b; opacity: 0.9; }
        [data-testid="stChatInput"] input:focus { box-shadow: none !important; }
        [data-testid="stChatInput"] button {
          background: #f8fafc !important; color: #334155 !important;
          border: 1px solid #e2e8f0 !important; width: 44px; height: 44px;
          border-radius: 999px; display: grid; place-items: center; margin-left: 6px;
          box-shadow: 0 1px 3px rgba(16,24,40,0.08) !important;
        }
        [data-testid="stChatInput"] button:hover { background: #ffffff !important; border-color: #cbd5e1 !important; }
        [data-testid="stChatInput"] button svg, [data-testid="stChatInput"] button path { fill: #334155 !important; stroke: #334155 !important; }

        /* Chat messages – give them a light gray background so they pop */
        [data-testid="stChatMessage"] {
          border: 1px solid #e5e7eb;
          background: #f9fafb;
          border-radius: 12px;
          padding: 0.5rem 0.75rem;
          margin-bottom: 0.5rem;
          box-shadow: 0 1px 3px rgba(16,24,40,0.04);
          color: #0f172a !important;  /* 👈 readable text */
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0; }

        .badge {
          display: inline-block; padding: 0.35rem 0.6rem; border-radius: 999px;
          background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
          font-weight: 600; margin: 0.2rem 0.2rem 0 0;
        }

        .totals-card {
          background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
          padding: 0.75rem; box-shadow: 0 1px 3px rgba(16,24,40,0.05);
          color: #0f172a;
        }

        .hero {
          border-radius: 16px; padding: 1.2rem 1.2rem 0.8rem; margin: 1.0rem 0 0.8rem;
          background: #ffffff;
          border: 1px solid #e5e7eb;
        }
        .hero-row { display:flex; align-items:center; gap:12px; }
        .hero-icon { width:40px; height:40px; border-radius:999px; background:#e5e7eb; display:grid; place-items:center; color:#334155; font-size:22px; }
        .hero h1 { margin: 0; font-size: 1.6rem; color:#111827 !important; background:none !important; -webkit-background-clip:initial; }
        .hero p { margin: 0.35rem 0 0; color: #6b7280; }

        .upload-card {
          border-radius: 14px; padding: 1rem; border: 1px dashed #e5e7eb;
          background: linear-gradient(180deg, #ffffffaa, #ffffff80);
          text-align: center;
          animation: subtlePulse 3s ease-in-out infinite;
          color: #0f172a;
        }

        @keyframes subtlePulse {
          0%,100%{ box-shadow: 0 0 0 rgba(100,116,139,0.0);}
          50%{ box-shadow: 0 10px 20px rgba(100,116,139,0.12);}
        }

        .macro-bars { margin-top: 0.5rem; }
        .bar { height: 10px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin: 6px 0; }
        .bar > span { display: block; height: 100%; border-radius: 999px; }
        .bar .protein { background: linear-gradient(90deg, #22c55e, #86efac); }
        .bar .fat { background: linear-gradient(90deg, #f59e0b, #fde68a); }
        .bar .carbs { background: linear-gradient(90deg, #10b981, #6ee7b7); }

        .chip {
          display:inline-block; padding: 0.25rem 0.55rem; border-radius: 999px;
          background:#f1f5f9; border:1px solid #e2e8f0; margin-right:6px;
          font-weight:600; color:#334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def reset_session() -> None:
    st.session_state.chat_session = None
    st.session_state.pipeline_result = None
    st.session_state.messages = []
    st.session_state.uploaded_image = None
    st.session_state.uploaded_mime_type = None
    st.session_state.did_upload = False


def render_sidebar() -> None:
    # Sidebar intentionally minimal per request (no upload here)
    st.sidebar.markdown("")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-row">
            <div class="hero-icon">🍽️</div>
            <h1>Food Nutrition Agent</h1>
          </div>
          <p>Upload a meal photo to estimate nutrition and ask follow-up questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary() -> None:
    result = st.session_state.pipeline_result
    if not result:
        st.info("No analysis result available yet.")
        return

    # 🔹 Show uploaded image (if available)
    if st.session_state.get("uploaded_image") is not None:
        st.subheader("Uploaded Image")
        st.image(
            st.session_state.uploaded_image,
            caption="Meal photo you uploaded",
            use_container_width=True,
        )

    st.subheader("Description")

    # Try a few likely locations for description text
    dish = (result.get("dish") or {}) if isinstance(result.get("dish"), dict) else {}
    description = (
        dish.get("description")
        or result.get("description")
        or dish.get("dish_name")
        or "No description was generated for this image."
    )

    # Optionally include some tags if you like
    cuisine = dish.get("cuisine") or ""
    category = dish.get("category") or ""

    # main description text
    st.markdown(description)

    # small tags under the description
    tags_html_parts = []
    if category:
        tags_html_parts.append(f"<span class='chip'>#{category}</span>")
    if cuisine:
        tags_html_parts.append(f"<span class='chip'>#{cuisine}</span>")

    if tags_html_parts:
        st.markdown("".join(tags_html_parts), unsafe_allow_html=True)

def render_chat() -> None:
    st.subheader("Chat with Nutrition Agent")
    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about this meal...")
    if user_input:
        if not st.session_state.chat_session:
            st.warning("Please upload and analyze an image first.")
            return
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chat_session.ask(user_input)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="upload-card">
          <div style="font-size:48px;line-height:1">📤</div>
          <h3 style="margin:6px 0 6px 0">Upload a meal photo to begin</h3>
          <div style="color:#475569">Drag & drop or choose a JPG/PNG/WEBP. We'll analyze it and you can chat about the results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload a meal photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        mime_type = uploaded.type or "image/jpeg"

        # 🔹 store the image in session
        st.session_state.uploaded_image = image_bytes
        st.session_state.uploaded_mime_type = mime_type

        with st.spinner("Analyzing image..."):
            result = process_image_bytes(image_bytes, mime_type=mime_type)

        st.session_state.pipeline_result = result
        st.session_state.chat_session = create_chat_session_from_result(result)
        st.session_state.messages = []
        st.session_state.did_upload = True
        st.rerun()

    # Bottom-left reset button under the upload area
    left, _ = st.columns([1, 6])
    with left:
        st.button("Reset", on_click=reset_session)

def main() -> None:
    st.set_page_config(page_title="Food Nutrition Chat", page_icon="🍽️", layout="centered", initial_sidebar_state="collapsed")
    init_env()
    ensure_api_key()
    ensure_session_state()
    inject_css()
    render_sidebar()
    render_header()
    if st.session_state.pipeline_result is None:
        render_empty_state()
    else:
        render_result_summary()
        render_chat()


if __name__ == "__main__":
    main()


