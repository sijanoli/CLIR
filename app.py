import streamlit as st
import pandas as pd
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import MultifieldParser
from deep_translator import GoogleTranslator
from langdetect import detect
import os

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Cross-Language Information Retrieval", page_icon="🌐", layout="wide")

st.title("🌐 Cross-Language Information Retrieval (CLIR) Demo")
st.write("Search English articles using queries in any language — powered by translation + classical IR (Whoosh BM25).")

# -----------------------------
# DATA UPLOAD SECTION
# -----------------------------
st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV with 'title' and 'content' columns):", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna().head(2000)  # limit to 2000 docs for demo
    st.success(f"✅ Loaded {len(df)} records successfully!")

    # -----------------------------
    # BUILD INDEX (Whoosh)
    # -----------------------------
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    schema = Schema(id=ID(stored=True), title=TEXT(stored=True), content=TEXT(stored=True))
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    for i, row in df.iterrows():
        writer.add_document(id=str(i), title=row["title"], content=row["content"])
    writer.commit()
    st.sidebar.success("Whoosh index created successfully ✅")

    # -----------------------------
    # QUERY SECTION
    # -----------------------------
    st.subheader("🔍 Try a Query")

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
            translated_query = query_text  # fallback to same text if translation fails

        st.write(f"**Detected Language:** `{lang}`")
        st.write(f"**Translated Query:** `{translated_query}`")

        # Search in index
        with ix.searcher() as searcher:
            parser = MultifieldParser(["title", "content"], schema=ix.schema)
            parsed_query = parser.parse(translated_query)
            results = searcher.search(parsed_query, limit=5)

            st.subheader("📄 Top Search Results")
            if len(results) == 0:
                st.warning("No results found.")
            else:
                for r in results:
                    st.markdown(f"### {r['title']}")
                    st.write(r["content"][:250] + "...")
                    st.divider()
else:
    st.info("👈 Upload a CSV file to begin. (Must have 'title' and 'content' columns.)")

st.sidebar.caption("Developed by [Sijan Oli] —  IR Project")
