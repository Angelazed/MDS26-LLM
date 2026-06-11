"""
Streamlit UI for the F1 RAG chatbot.

Run with:  streamlit run app.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Knowledge Assistant",
    page_icon="🏎️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Pipeline (cached so it loads only once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading F1 knowledge base …")
def get_pipeline():
    from rag_pipeline import build_pipeline
    return build_pipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏎️ F1 RAG Assistant")
    st.markdown(
        """
        Ask anything about Formula One:
        - 🏆 Drivers & championships
        - 🏁 Teams & constructors
        - 🔧 Technical regulations
        - 🗺️ Circuits
        - 📖 F1 history
        """
    )
    st.divider()

    top_k = st.slider("Chunks retrieved (top-k)", min_value=1, max_value=10, value=5)
    show_sources = st.checkbox("Show source passages", value=True)

    st.divider()
    st.caption("Corpus: 41 Wikipedia articles · ~4,300 chunks")
    st.caption("Embeddings: all-MiniLM-L6-v2 (local)")
    st.caption("LLM: Llama 3.1 via Ollama (local)")

# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("Formula One Knowledge Assistant")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📄 Retrieved sources"):
                for s in msg["sources"]:
                    label = f"**{s['title']}**"
                    if s["section"]:
                        label += f" › {s['section']}"
                    st.markdown(f"{label} — relevance score: `{s['score']}`  \n[Wikipedia]({s['url']})")

# Chat input
if not st.session_state.messages:
    st.markdown("### Try asking:")
    examples = [
        "What is the DRS system in Formula One?",
        "Who is Lando Norris and which team does he race for?",
        "Tell me about the Circuit de Monaco.",
        "How many championships did Michael Schumacher win?",
        "What type of engines are used in Formula One?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

if prompt := st.chat_input("Ask about Formula 1 …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking …_")
        try:
            pipeline = get_pipeline()
            pipeline._top_k = top_k
            result  = pipeline.ask(prompt)
            answer  = result["answer"]
            sources = result["sources"]

            placeholder.markdown(answer)

            if show_sources and sources:
                with st.expander("📄 Retrieved sources"):
                    for s in sources:
                        label = f"**{s['title']}**"
                        if s["section"]:
                            label += f" › {s['section']}"
                        st.markdown(f"{label} — relevance score: `{s['score']}`  \n[Wikipedia]({s['url']})")

        except Exception as e:
            answer  = f"❌ Error: {e}"
            sources = []
            placeholder.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

