"""
IUB Assistant — Streamlit chat app for the RAG pipeline built in Rag_Iub.ipynb.

Setup:
1. Create a `.env` file in the same folder as this script with:
       DB_CONNECTION=postgresql+psycopg2://user:password@host/dbname?sslmode=require
       GROQ_API_KEY=your_groq_api_key
   (Use a NEW Groq key and a NEW Neon password — rotate any credentials that
   were ever pasted into a notebook, chat, or shared file.)
2. Make sure the "IUB_collection" already exists in Postgres (i.e. you've
   already run the ingestion notebook to embed and store the PDFs).
3. Run with:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_CONNECTION = os.getenv("DB_CONNECTION")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "IUB_collection"

st.set_page_config(page_title="IUB Assistant", page_icon="🎓", layout="centered")

if not DB_CONNECTION or not GROQ_API_KEY:
    st.error(
        "Missing credentials. Create a `.env` file next to app.py with "
        "`DB_CONNECTION` and `GROQ_API_KEY` set (see the comment block at "
        "the top of app.py)."
    )
    st.stop()

PROMPT = PromptTemplate(
    template="""
    You are a helpful assistant for The Islamia University of Bahawalpur (IUB).
    Answer the student's question using ONLY the context provided below, which is
    retrieved from official IUB documents (admission regulations, faculties, and
    departments).

    Rules:
    - If the answer is not present in the context, say "I don't have that information
      in the available documents" — do not guess or use outside knowledge.
    - If the question involves dates, fees, seat numbers, or age limits, quote the
      figures exactly as given in the context.
    - Keep answers concise and direct; use bullet points only if the context itself
      is a list.
    - If multiple context chunks seem relevant but conflict, mention the discrepancy
      rather than picking one silently.

    Context:
    {context}

    Question: {question}

    Answer:
    """,
    input_variables=["context", "question"],
)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_retriever():
    vector_store = PGVector(
        embeddings=load_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=256)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str):
    retriever = load_retriever()
    llm = load_llm()
    parser = StrOutputParser()

    retrieved_docs = retriever.invoke(question)
    context_text = format_docs(retrieved_docs)
    final_prompt = PROMPT.invoke({"context": context_text, "question": question})
    response = llm.invoke(final_prompt)
    answer = parser.invoke(response)

    sources, seen = [], set()
    for doc in retrieved_docs:
        file_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_label", "?")
        key = (file_name, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": file_name, "page": page})

    return answer, sources


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for src in sources:
            st.markdown(f"- **{src['file']}**, page {src['page']}")


st.title("🎓 IUB Assistant")
st.caption(
    "Ask about admission regulations, faculties, and departments at "
    "The Islamia University of Bahawalpur. Answers are based only on the "
    "official documents that were indexed into this assistant."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about IUB admissions or faculties...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, sources = answer_question(question)
            except Exception as exc:
                answer, sources = f"Something went wrong: {exc}", []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves relevant passages from IUB's admission "
        "regulations and faculty/department PDFs using a PGVector-backed "
        "retriever, then answers with Llama 3.1 8B via Groq."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
