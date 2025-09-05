# create_vector.py
import json
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

CACHE_FILE = "news_cache.json"  # adjust to your file

def create_vector_db(base_url="http://host.docker.internal:11434", chunk_size=500, chunk_overlap=50):
    """
    Loads cached JSON articles, filters out anti-scraping content,
    splits into chunks, and creates a FAISS vector store.
    """
    # Load cached articles
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        articles_json = json.load(f)
    df = pd.DataFrame(articles_json)

    # Combine title, description, content with anti-scraping filter
    def combine_article_text(row):
        title = row.get("title") or ""
        description = row.get("description") or ""
        content = row.get("content") or ""
        combined = f"{title} {description} {content}"

        # Filter for anti-scraping/security messages
        block_phrases = [
            "security service", "captcha", "online attacks",
            "action you just performed", "Cloudflare"
        ]
        if any(phrase.lower() in combined.lower() for phrase in block_phrases):
            return ""  # skip this article
        return combined

    df["combined_text"] = df.apply(combine_article_text, axis=1)
    df = df[df["combined_text"].str.strip() != ""]  # remove empty articles

    # Split articles into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for text in df["combined_text"]:
        chunks = text_splitter.split_text(text)
        all_chunks.extend([c for c in chunks if c.strip() != ""])  # remove empty chunks

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="tinyllama",
        base_url=base_url
    )

    # Create FAISS vector store
    vector_db = FAISS.from_texts(all_chunks, embeddings)
    print(f"FAISS vector store created with {vector_db.index.ntotal} vectors.")
    return vector_db

if __name__ == "__main__":
    create_vector_db()
