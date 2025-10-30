import streamlit as st
import pandas as pd
import os
import json
import kagglehub

from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import MultifieldParser
from deep_translator import GoogleTranslator
from langdetect import detect

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="CLIR with Kaggle News Dataset", page_icon="🌐", layout="wide")
st.title("🌐 Cross-Language Information Retrieval (CLIR) — News Search")
st.write("Search the Kaggle **News Category Dataset** in any language. Query is translated → searched in English → top matching articles are shown.")

# -----------------------------
# STEP 1: LOAD DATA FROM KAGGLE
# -----------------------------
st.sidebar.header("📡 Dataset Loading")

st.write("⏳ Downloading dataset from Kaggle...")
try:
    # Download latest dataset version
    path = kagglehub.dataset_download("rmisra/news-category-dataset")
    data_file = os.path.join(path, "News_Category_Dataset_v2.json")

    # Load dataset
    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)[["headline", "short_description"]].dropna().head(2000)
    df = df.rename(columns={"headline": "title", "short_description": "content"})

    st.success(f"✅ Loaded {len(df)} news articles from Kaggle dataset.")

except Exception as e:
    st.error("❌ Error downloading dataset. Check KaggleHub or internet connection.")
    st.stop()

# -----------------------------
# STEP 2: BUILD INDEX (WHOOSH)
# -----------------------------
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

schema = Schema(id=ID(stored=True), title=TEXT(stored=True), content=TEXT(stored=True))
ix = index.create_in("indexdir", schema)
writer = ix.writer()

for i, row in df.iterrows():
    writer.add_document(id=str(i), title=row["title"], content=row["content"])
writer.commit()
st.sidebar.success("Index built successfully ✅")

# -----------------------------
# STEP 3: QUERY SECTION
# -----------------------------
st.subheader("🔍 Search Query")

query_text = st.text_input("Enter query (in any language):", "")

if query_text:
    # Detect language
    try:
        lang = detect(query_text)
    except:
        lang = "unknown"

    # Translate to English
    try:
        translated_query = GoogleTranslator(source=lang, target='en').translate(query_text)
    except:
        translated_query = query_text  # fallback

    st.write(f"**Detected Language:** `{lang}`")
    st.write(f"**Translated Query:** `{translated_query}`")

    # Search index
    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "content"], schema=ix.schema)
        parsed_query = parser.parse(translated_query)
        results = searcher.search(parsed_query, limit=5)

        st.subheader("📄 Top Results")
        if len(results) == 0:
            st.warning("No results found.")
        else:
            for r in results:
                st.markdown(f"### {r['title']}")
                st.write(r["content"][:250] + "...")
                st.divider()

st.sidebar.caption("Developed by [Your Name] — CSE466 IR Project")
