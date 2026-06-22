# 🎓 IUB Assistant — RAG Chatbot for The Islamia University of Bahawalpur

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about **admissions, faculties, and departments** at The Islamia University of Bahawalpur (IUB). It retrieves relevant passages from official IUB PDF documents and generates grounded answers using **Llama 3.1 8B** via Groq — never hallucinating beyond what the documents say.

---

## 🏗️ Architecture

```
PDF Documents (DOCS/)
        │
        ▼
DirectoryLoader + PyPDFLoader          ← Load PDFs
        │
        ▼
RecursiveCharacterTextSplitter         ← Chunk (1000 chars, 150 overlap)
        │
        ▼
HuggingFace all-MiniLM-L6-v2          ← Embed chunks
        │
        ▼
PGVector (Neon PostgreSQL)             ← Store & index embeddings
        │
   [at query time]
        │
        ▼
MMR Retriever (k=4, fetch_k=20)       ← Retrieve relevant chunks
        │
        ▼
PromptTemplate + ChatGroq (Llama 3.1) ← Generate answer
        │
        ▼
Streamlit Chat UI                      ← Display answer + sources
```

---

## 📁 Project Structure

```
IUB_RAG_Chatbot/
│
├── DOCS/                          # IUB PDF source documents
│   ├── IUB_Admission_Regulations_2015.pdf
│   └── IUB_Faculties_and_Departments.pdf
│
├── Rag_Iub.ipynb                  # Ingestion notebook (chunk → embed → store)
├── app.py                         # Streamlit chat application
├── .env                           # Credentials (not committed)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/IUB-RAG-Chatbot.git
cd IUB-RAG-Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv myvenv
# Windows
myvenv\Scripts\activate
# macOS/Linux
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Create a `.env` file in the project root:

```env
DB_CONNECTION=postgresql+psycopg2://user:password@host/dbname?sslmode=require
GROQ_API_KEY=your_groq_api_key
```

> ⚠️ **Never commit your `.env` file.** Add it to `.gitignore`.  
> Get a free Groq API key at [console.groq.com](https://console.groq.com).  
> A free Neon PostgreSQL database works at [neon.tech](https://neon.tech).

### 5. Add your PDF documents

Place all IUB PDF files inside a `DOCS/` folder in the project root.

### 6. Run the ingestion notebook

Open and run `Rag_Iub.ipynb` end-to-end. This will:
- Load and chunk all PDFs in `DOCS/`
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Store them in your Neon PostgreSQL database as the `IUB_collection`

### 7. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Document Loading | LangChain `DirectoryLoader` + `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | `PGVector` on Neon (serverless PostgreSQL) |
| Retrieval Strategy | MMR (Maximal Marginal Relevance) |
| LLM | Llama 3.1 8B Instant via Groq |
| Orchestration | LangChain |
| Frontend | Streamlit |

---

## 💬 Features

- **Grounded answers only** — the LLM is instructed to answer strictly from retrieved context and say "I don't have that information" rather than guess.
- **Source citations** — every answer shows which PDF file and page the answer came from.
- **MMR retrieval** — avoids redundant chunks by maximizing diversity among the top-k results.
- **Conflict detection** — the prompt instructs the model to surface conflicting information across chunks rather than silently pick one.
- **Persistent chat history** — the conversation is preserved within the session with a one-click clear button.
- **Cached resources** — embeddings, retriever, and LLM are cached with `@st.cache_resource` for fast repeated queries.

---

## 📸 Demo

> Ask questions like:
> - *"How can I get admission in the Software Engineering department?"*
> - *"What are the reserved seats for students from AJ&K?"*
> - *"What engineering programs does UCET offer?"*

---

## 📋 Requirements

```
langchain
langchain-community
langchain-huggingface
langchain-postgres
langchain-groq
langchain-core
langchain-text-splitters
streamlit
python-dotenv
psycopg2-binary
sentence-transformers
pypdf
```

---

## 🔒 Security Notes

- Rotate any API keys or database passwords that were ever pasted directly into a notebook, chat, or shared file.
- Use environment variables (`.env`) for all credentials — never hardcode them.
- Add `.env` and `myvenv/` to your `.gitignore`.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [The Islamia University of Bahawalpur](https://www.iub.edu.pk) for the source documents.
- [Groq](https://groq.com) for ultra-fast LLM inference.
- [Neon](https://neon.tech) for serverless PostgreSQL with pgvector support.
- [LangChain](https://www.langchain.com) for the RAG framework.
