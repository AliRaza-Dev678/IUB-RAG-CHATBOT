# app.py
# Streamlit chat UI for IUB RAG Assistant.
# Run: python -m streamlit run app.py

import os
import streamlit as st
from dotenv import load_dotenv
from rag import get_embeddings, get_retriever, get_llm, get_chain, get_answer

load_dotenv()

# ---------- Page setup ----------
st.set_page_config(page_title="IUB Assistant", page_icon="🎓", layout="centered")
st.title("🎓 IUB Assistant")
st.caption("Ask anything about IUB — admissions, programs, fees, faculties, and more.")

# ---------- Check .env keys ----------
missing = [k for k in ["DB_CONNECTION", "GROQ_API_KEY", "DIRECT_URL"] if not os.getenv(k)]
if missing:
    st.error(f"Missing in .env: {', '.join(missing)}")
    st.stop()

# ---------- Load RAG components (cached so they load only once) ----------
@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    embeddings = get_embeddings()
    retriever  = get_retriever(embeddings)
    llm        = get_llm()
    chain      = get_chain(retriever, llm)
    return retriever, chain

retriever, chain = load_resources()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for url in msg["sources"]:
                    st.markdown(f"- [{url}]({url})")

# ---------- Handle new question ----------
question = st.chat_input("Ask a question about IUB...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                answer, sources = get_answer(question, retriever, chain)
            except Exception as e:
                answer  = f"Something went wrong: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for url in sources:
                    st.markdown(f"- [{url}]({url})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions about IUB using official website content stored in a Postgres vector database.")
    st.divider()
    st.markdown("**Model:** `llama-3.1-8b-instant`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Retrieval:** MMR top 4 chunks")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()