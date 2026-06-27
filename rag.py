# rag.py
# Core RAG logic: embeddings, retriever, LLM, prompt, chain.
# This file is imported by app.py — do not run directly.

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

COLLECTION_NAME = "IUB_collection"

# ---------- Embeddings ----------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------- Retriever ----------
def get_retriever(embeddings):
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.getenv("DB_CONNECTION"),
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

# ---------- LLM ----------
def get_llm():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=256,
    )

# ---------- Prompt ----------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant for The Islamia University of Bahawalpur (IUB).
    Answer the student's question using ONLY the context provided below, which is
    retrieved from official IUB documents (admission regulations, faculties, and departments).

    Rules:
    - If the answer is not in the context, say "I don't have that information in the available documents."
    - Do not guess or use outside knowledge.
    - If the question involves dates, fees, seat numbers, or age limits, quote figures exactly.
    - Keep answers concise; use bullet points only if the context itself is a list.

    Context:
    {context}

    Question: {question}

    Answer:
    """,
)

# ---------- Chain ----------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def get_chain(retriever, llm):
    parallel_chain = RunnableParallel({
        "context" : retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    return parallel_chain | prompt | llm | StrOutputParser()

# ---------- Answer function ----------
def get_answer(question, retriever, chain):
    answer = chain.invoke(question)

    retrieved_docs = retriever.invoke(question)
    sources, seen = [], set()
    for doc in retrieved_docs:
        url = doc.metadata.get("source", "unknown")
        if url not in seen:
            seen.add(url)
            sources.append(url)

    return answer, sources