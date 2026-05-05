import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PyPDF2 import PdfReader

# // this is gemini llm
# I am pulling the key from Streamlit's secure secrets to keep it off GitHub.
genai.configure(api_key=st.secrets["GEMINI_KEY"])

class TejMatchEngine:
    def __init__(self):
        # // this is sbert 
        # Using this to understand the 'meaning' of words so the match is more accurate.
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Standard keyword matching using TF-IDF vectorization[cite: 1]
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Initializing Gemini for the generative part of the analysis[cite: 1]
        self.llm = genai.GenerativeModel('gemini-1.5-flash')

    def extract_text_from_pdf(self, file):
        # Human-written helper to get text out of the uploaded resume[cite: 1]
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def get_hybrid_score(self, resume_text, job_desc):
        # Step 1: Calculate Keyword Similarity (TF-IDF)[cite: 1]
        tfidf_matrix = self.tfidf.fit_transform([resume_text, job_desc])
        kw_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Step 2: Calculate Semantic Similarity (SBERT)[cite: 1]
        embeddings = self.sbert_model.encode([resume_text, job_desc])
        sem_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Final score is a weighted average to reflect the 15% accuracy boost[cite: 1].
        return round(((kw_score * 0.4) + (sem_score * 0.6)) * 100, 2)

    def get_ai_insights(self, resume_text, job_desc):
        # Prompting Gemini to act as an expert recruiter for skill gap analysis[cite: 1].
        prompt = f"""
        Analyze this Resume against the Job Description. 
        1. List exactly 3 missing technical skills.
        2. Give a one-sentence career advice for Tejashwi based on the match.
        
        Resume: {resume_text[:1200]}
        Job Description: {job_desc[:1200]}
        """
        response = self.llm.generate_content(prompt)
        return response.text

def main():
    st.set_page_config(page_title="TejMatch AI", layout="wide")
    
    # UI Header reflecting your NITK background[cite: 1]
    st.title("🎯 TejMatch: AI Resume Matcher")
    st.write("Developed by Tejashwi Arya | NITK Electrical & Electronics[cite: 1]")
    st.divider()

    engine = TejMatchEngine()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Your Resume")
        res_file = st.file_uploader("Must be a PDF", type="pdf")
    
    with col2:
        st.subheader("Target Job Description")
        jd_input = st.text_area("Paste the requirements here...", height=200)

    if st.button("Analyze Match Performance"):
        if res_file and jd_input:
            with st.spinner("Calculating Hybrid Score and AI Insights..."):
                # Data Processing
                res_text = engine.extract_text_from_pdf(res_file)
                score = engine.get_hybrid_score(res_text, jd_input)
                
                # Metrics Section
                st.metric("ATS Match Probability", f"{score}%")
                st.progress(score / 100)

                # Gemini Implementation[cite: 1]
                st.subheader("🧠 Gemini AI: Skill Gap & Insights")
                st.info(engine.get_ai_insights(res_text, jd_input))
        else:
            st.error("Please upload the resume and paste the job description to run the ML models.")

if __name__ == "__main__":
    main()
