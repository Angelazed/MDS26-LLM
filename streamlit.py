# Streamlit

import streamlit as st

try:
    from rag_pipeline import answer_question

except ImportError:

    def answer_question(query, top_k=5):
        return {
            "answer": (
                "RAG pipeline not connected yet.\n\n"
                "Make sure rag_pipeline.py contains:\n"
                "answer_question(query, top_k)"
            ),
            "sources": [],
            "contexts": []
        }


# Configuration of Streamlit
st.set_page_config(
    page_title="Formula1 RAG",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ Formula1 RAG Chatbot")

st.markdown(
    """
Ask questions about Formula One, including:

- Drivers
- Circuits
- Regulations
- Engines
- History
"""
)

st.divider()

st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Number of retrieved documents",
    min_value=1,
    max_value=10,
    value=5
)

show_sources = st.sidebar.checkbox(
    "Show sources",
    value=True
)

show_context = st.sidebar.checkbox(
    "Show retrieved context",
    value=False
)

question = st.text_input(
    "Enter your Formula One question:"
)
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            response = answer_question(question, top_k=top_k)

        st.subheader("Answer")
        st.write(response.get("answer", "No answer generated."))

        if show_sources:                          # ← indented inside here
            sources = response.get("sources", [])
            st.subheader("Sources")
            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No sources available.")
        

if show_sources:

            sources = response.get(
                "sources",
                []
            )

            st.subheader("Sources")

            if sources:

                for source in sources:
                    st.write(f"- {source}")

            else:
                st.write(
                    "No sources available."
                )

st.divider()

st.caption(
    "⚠️ This system uses a Retrieval-Augmented Generation (RAG) approach. Although answers are grounded in retrieved sources, occasional errors or inaccuracies may still occur."
)