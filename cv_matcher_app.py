import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from tqdm import tqdm
import time

# --- Configure OpenAI ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def compute_tfidf_similarity(cv_texts, jd_text):
    corpus = [jd_text] + cv_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    cv_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(cv_vectors, jd_vector).flatten()
    return scores

def get_embedding(text, model="text-embedding-3-small", retries=3):
    for attempt in range(retries):
        try:
            response = openai.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except openai.RateLimitError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise

def compute_openai_similarity(cv_texts, jd_text):
    jd_embed = get_embedding(jd_text)
    cv_embeds = []
    for text in tqdm(cv_texts):
        embed = get_embedding(text)
        cv_embeds.append(embed)
        time.sleep(1)  # avoid rate limits

    from numpy import dot
    from numpy.linalg import norm

    scores = [dot(cv, jd_embed) / (norm(cv) * norm(jd_embed)) for cv in cv_embeds]
    return scores

# --- Streamlit UI ---
st.title("The Geezers CV Matcher App")
st.write("Upload a job description and multiple CVs to find the best matches.")

# Upload Job Description
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Upload CVs
cv_files = st.file_uploader("Upload CVs (PDFs)", type=["pdf"], accept_multiple_files=True)

# Matching Method
method = st.radio("Choose Matching Method", ["TF-IDF", "OpenAI Embeddings"], index=1)

if jd_file and cv_files:
    with st.spinner("Processing..."):
        jd_text = extract_text_from_pdf(jd_file)
        cv_texts = [extract_text_from_pdf(cv) for cv in cv_files]
        cv_names = [cv.name for cv in cv_files]

        if method == "TF-IDF":
            scores = compute_tfidf_similarity(cv_texts, jd_text)
        else:
            scores = compute_openai_similarity(cv_texts, jd_text)

        results = pd.DataFrame({
            "CV File": cv_names,
            "Match Score (%)": (pd.Series(scores) * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        st.success("Matching complete!")
        st.subheader("Top Matches")
        st.dataframe(results)

        # Download CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "cv_match_results.csv", "text/csv")
