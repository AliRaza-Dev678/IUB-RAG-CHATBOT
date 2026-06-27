# ingest.py
# Run this ONCE to scrape IUB website and store data in database.
# Usage: python ingest.py

import os
import psycopg2
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# ---------- Step 1: Clear old data ----------
print("Clearing old database tables...")
conn = psycopg2.connect(os.getenv("DIRECT_URL"))
conn.autocommit = True
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding;")
cur.execute("DROP TABLE IF EXISTS langchain_pg_collection;")
cur.close()
conn.close()
print("Done.")

# ---------- Step 2: Load data from IUB website ----------
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

# Loader 1: crawl entire site
print("\nCrawling IUB website...")
crawler = RecursiveUrlLoader(
    url="https://www.iub.edu.pk",
    max_depth=2,
    extractor=clean_html,
    prevent_outside=True,
)
crawled_docs = crawler.load()
print(f"Crawler got: {len(crawled_docs)} pages")

# Loader 2: load specific important pages
specific_urls = [
    "https://www.iub.edu.pk/admissions",
    "https://www.iub.edu.pk/fee-structure",
    "https://www.iub.edu.pk/faqs",
    "https://www.iub.edu.pk/academicprograms/undergraduate-programs",
    "https://www.iub.edu.pk/academicprograms/ms-programs",
    "https://www.iub.edu.pk/academicprograms/phd-programs",
    "https://www.iub.edu.pk/eligibility-criteria",
    "https://www.iub.edu.pk/scholarships",
    "https://www.iub.edu.pk/contact",
]

print("Loading specific pages...")
specific_docs = AsyncHtmlLoader(specific_urls).load()
specific_docs = Html2TextTransformer().transform_documents(specific_docs)
print(f"Specific pages got: {len(specific_docs)} pages")

# Combine both
all_docs = crawled_docs + specific_docs
print(f"Total documents: {len(all_docs)}")

# ---------- Step 3: Split into chunks ----------
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# ---------- Step 4: Create embeddings ----------
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ---------- Step 5: Store in PGVector ----------
print("Storing in database (this takes a few minutes)...")
PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="IUB_collection",
    connection=os.getenv("DB_CONNECTION"),
)

print(f"\nDone! {len(chunks)} chunks stored in database.")
print("Now run:  python -m streamlit run app.py")