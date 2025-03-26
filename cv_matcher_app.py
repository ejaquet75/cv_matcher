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
import re

# --- Configure OpenAI ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def truncate_text(text, max_tokens=1500):
    words = text.split()
    return " ".join(words[:max_tokens])

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
                time.sleep(2 ** attempt)
            else:
                raise

def compute_openai_similarity(cv_texts, jd_text):
    jd_text = truncate_text(jd_text)
    jd_embed = get_embedding(jd_text)
    cv_embeds = []
    for text in tqdm(cv_texts):
        embed = get_embedding(truncate_text(text))
        cv_embeds.append(embed)
        time.sleep(1)  # avoid rate limits

    from numpy import dot
    from numpy.linalg import norm

    scores = [dot(cv, jd_embed) / (norm(cv) * norm(jd_embed)) for cv in cv_embeds]
    return scores

# --- Estimate Years of Experience ---
def estimate_experience_years(text):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    years = sorted(set(int(y) for y in years))
    return max(0, (years[-1] - years[0])) if len(years) >= 2 else 0

# --- Extract Keywords from Job Description ---
def extract_keywords_from_jd(text, top_n=10):
    # Clean the text (remove special characters and multiple spaces)
    clean_text = ' '.join(text.split())
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([clean_text])
    feature_array = vectorizer.get_feature_names_out()
    return feature_array

# --- Streamlit UI ---
st.title("Geezer CV Matcher App")
st.write("Upload a job description and multiple CVs to find the best matches.")

# Upload Job Description
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Upload CVs
cv_files = st.file_uploader("Upload CVs (PDFs)", type=["pdf"], accept_multiple_files=True)

# Matching Method
method = st.radio("Choose Matching Method", ["AI-Powered Match", "TF-IDF"], index=0)

# Weighted scoring sliders
st.sidebar.header("🔧 Scoring Weights")
skill_weight = st.sidebar.slider("Keyword Matching Weight", 0.0, 1.0, 1.0, 0.1)

# Display keywords from the Job Description once it is uploaded
if jd_file:
    with st.spinner("Processing..."):
        jd_text = extract_text_from_pdf(jd_file)
        if len(jd_text.split()) > 1500:
            st.warning("Job description is long — truncating to 1500 words for semantic matching.")

        # Extract top keywords from JD
        top_keywords = extract_keywords_from_jd(jd_text)

        # Allow the user to remove unnecessary keywords
        keywords_to_remove = []
        for keyword in top_keywords:
            # Create a checkbox with 'X' for removal
            remove_keyword = st.checkbox(f"Remove {keyword}", key=keyword)
            if remove_keyword:
                keywords_to_remove.append(keyword)

        # Update the list of keywords after removal
        updated_keywords = [keyword for keyword in top_keywords if keyword not in keywords_to_remove]

        # Display updated keywords in the sidebar with a grey background
        keyword_tags = " ".join([f"<span style='background-color:#d9d9d9; padding:4px 8px; margin-right:4px; border-radius:6px;'>{keyword}</span>" for keyword in updated_keywords])
        st.sidebar.markdown(f"<div style='line-height:2.2'>{keyword_tags}</div>", unsafe_allow_html=True)

if jd_file and cv_files:
    with st.spinner("Processing..."):
        cv_texts = [extract_text_from_pdf(cv) for cv in cv_files]
        cv_names = [cv.name for cv in cv_files]

        # Extract years of experience for each CV
        experience_scores = [estimate_experience_years(text) for text in cv_texts]

        # Filter out the removed keywords from JD
        filtered_jd_text = ' '.join([word for word in jd_text.split() if word not in keywords_to_remove])

        if method == "TF-IDF":
            semantic_scores = compute_tfidf_similarity(cv_texts, filtered_jd_text)
        elif method == "AI-Powered Match":
            try:
                semantic_scores = compute_openai_similarity(cv_texts, filtered_jd_text)
            except Exception as e:
                st.error("⚠️ An error occurred while using AI-Powered Match.")
                with st.expander("🔧 Debug Tools"):
                    if st.button("Run Test Request"):
                        try:
                            test_response = openai.embeddings.create(
                                input=["Test if OpenAI key works."],
                                model="text-embedding-3-small"
                            )
                            st.success("✅ OpenAI API key is working!")
                        except openai.RateLimitError:
                            st.error("❌ Rate limit hit. Your API key may be over quota or too many requests were made.")
                        except openai.AuthenticationError:
                            st.error("❌ Invalid API key. Please check your secret.")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")
                    st.info("To view detailed usage, visit your OpenAI dashboard: https://platform.openai.com/account/usage")
                raise e

        shortlist_flags = []
        comments = []

        st.success("Matching complete!")
        st.subheader("Review & Shortlist")

        for i, name in enumerate(cv_names):
            with st.expander(f"{name} - Match Score: {(semantic_scores[i]*100):.2f}%"):
                is_shortlisted = st.checkbox(f"Shortlist {name}?", key=f"shortlist_{i}")
                comment = st.text_area("Comments", key=f"comment_{i}")
                shortlist_flags.append(is_shortlisted)
                comments.append(comment)

        # Apply weighted scoring
        final_scores = []
        for i in range(len(cv_names)):
            score = semantic_scores[i] * skill_weight
            final_scores.append(score)

        results = pd.DataFrame({
            "CV File": cv_names,
            "Match Score (%)": (pd.Series(final_scores) * 100).round(2),
            "Years of Experience": experience_scores,
            "Shortlisted": shortlist_flags,
            "Comments": comments
        }).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        st.subheader("Top Matches")
        st.dataframe(results)

        # Download CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Report", csv, "cv_match_results.csv", "text/csv")
