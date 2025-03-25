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

def extract_skills(text):
    # Example placeholder skill keywords
    keywords = ["python", "excel", "marketing", "sales", "sql", "data analysis"]
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower) / len(keywords)

def estimate_experience_years(text):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    years = sorted(set(int(y) for y in years))
    return max(0, (years[-1] - years[0])) if len(years) >= 2 else 0

# --- Streamlit UI ---
def extract_top_skills_from_jd(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_indices = tfidf_scores.argsort()[::-1][:top_n]
    return [feature_array[i] for i in top_indices if tfidf_scores[i] > 0]

st.title("Geezer CV Matcher App")
st.write("Upload a job description and multiple CVs to find the best matches.")

jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
cv_files = st.file_uploader("Upload CVs (PDFs)", type=["pdf"], accept_multiple_files=True)

method = st.radio("Choose Matching Method", ["AI-Powered Match", "TF-IDF"], index=0)

st.sidebar.header("🔧 Scoring Weights")
skill_weight = st.sidebar.slider("AI Keyword Match", 0.0, 1.0, 1.0, 0.1)
skills_match_weight = st.sidebar.slider("Skills Match Weight", 0.0, 1.0, 0.5, 0.1)
years_experience_weight = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.5, 0.1)

if jd_file:
    jd_text = extract_text_from_pdf(jd_file)
    top_skills = extract_top_skills_from_jd(jd_text)
    st.sidebar.markdown("### 🧠 Top JD Skills")

    # Let the user edit the skills
    selected_skills = st.sidebar.multiselect("Edit or remove detected skills:", top_skills, default=top_skills)
    new_skill = st.sidebar.text_input("Add a new skill:")
    if new_skill and new_skill.strip() and new_skill.lower() not in [s.lower() for s in selected_skills]:
        selected_skills.append(new_skill.strip())

    # Display skill tags
    skill_tags = " ".join([
        f"<span style='background-color:#c0c0c0; padding:4px 8px; margin-right:4px; border-radius:6px;'>{skill}</span>"
        for skill in selected_skills
    ])
    st.sidebar.markdown(f"<div style='line-height:2.2'>{skill_tags}</div>", unsafe_allow_html=True)

if jd_file and cv_files:
    with st.spinner("Processing..."):
        jd_text = extract_text_from_pdf(jd_file)
        if len(jd_text.split()) > 1500:
            st.warning("Job description is long — truncating to 1500 words for semantic matching.")

        cv_texts = [extract_text_from_pdf(cv) for cv in cv_files]
        cv_names = [cv.name for cv in cv_files]

        if method == "TF-IDF":
            semantic_scores = compute_tfidf_similarity(cv_texts, jd_text)
        elif method == "AI-Powered Match":
            try:
                semantic_scores = compute_openai_similarity(cv_texts, jd_text)
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

        shortlist_flags = [False for _ in cv_names]
        comments = ["" for _ in cv_names]

        skill_matches = [extract_skills(text) for text in cv_texts]
        experience_scores = [estimate_experience_years(text)/10 for text in cv_texts]  # normalize years

        final_scores = []
        for i in range(len(cv_names)):
            score = (
                semantic_scores[i] * skill_weight +
                skill_matches[i] * skills_match_weight +
                experience_scores[i] * years_experience_weight
            )
            final_scores.append(score)

        st.success("Matching complete!")
        st.subheader("Review & Shortlist")

        for i, name in enumerate(cv_names):
            with st.expander(f"{name} - Match Score: {(final_scores[i]*100):.2f}%"):
                shortlist_flags[i] = st.checkbox(f"Shortlist {name}?", key=f"shortlist_{i}")
                comments[i] = st.text_area("Comments", key=f"comment_{i}")

        results = pd.DataFrame({
            "CV File": cv_names,
            "Match Score (%)": (pd.Series(final_scores) * 100).round(2),
            "Skill Match Score": (pd.Series(skill_matches) * 100).round(1),
            "Years Experience": [int(e * 10) for e in experience_scores],
            "Shortlisted": shortlist_flags,
            "Comments": comments
        }).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        st.subheader("Top Matches")
        st.dataframe(results)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Report", csv, "cv_match_results.csv", "text/csv")
