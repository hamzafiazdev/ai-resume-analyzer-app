import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type="pdf")
job_desc = st.text_area("Paste Job Description")

if uploaded_file and job_desc:
    pdf = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf.pages:
        resume_text += page.extract_text()

    text = [resume_text, job_desc]

    cv = CountVectorizer()
    matrix = cv.fit_transform(text)

    similarity = cosine_similarity(matrix)[0][1]

    st.success(f"Match Score: {round(similarity*100,2)}%")
