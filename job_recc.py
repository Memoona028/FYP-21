# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:26:55 2024

@author: DELL
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re

# Ensure necessary NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('corpora/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("job_descriptions.csv")

job_df = load_data()

# Preprocessing
ps = PorterStemmer()

# Clean text function
def cleaning(txt):
    if pd.isnull(txt):  # Check for NaN values
        return ""
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)  # Remove special characters
    tokens = nltk.word_tokenize(txt.lower())
    stemming = [ps.stem(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(stemming)

# Sample dataset for processing
@st.cache_data
def sample_data():
    return job_df.sample(n=10000)  # Adjust sample size to 10,000 as per Jupyter notebook

job_df_sample = sample_data()

# Convert 'Job Posting Date' to datetime
job_df_sample['Job Posting Date'] = pd.to_datetime(job_df_sample['Job Posting Date'], errors='coerce')

# Columns to clean (matching the Jupyter notebook columns)
text_columns = ['Experience', 'Qualifications', 'Salary Range', 'location', 
                'Country', 'Work Type', 'Preference', 'Contact Person', 
                'Contact', 'Job Title', 'Role', 'Job Portal', 'skills', 
                'Job Description', 'Benefits', 'Responsibilities', 'Company']

# Apply cleaning
@st.cache_data
def preprocess_data():
    for col in text_columns:
        job_df_sample[col] = job_df_sample[col].astype(str).apply(cleaning)

    # Combine text for TF-IDF
    job_df_sample['clean_text'] = job_df_sample[text_columns].apply(lambda row: ' '.join(row), axis=1)

preprocess_data()

# Cache TF-IDF and similarity computation
@st.cache_data
def compute_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(job_df_sample['clean_text'])
    similarity = cosine_similarity(matrix)
    return similarity

# Store the similarity matrix in session state to avoid recomputing
if 'similarity' not in st.session_state:
    st.session_state.similarity = compute_similarity()

# Streamlit App
st.title("Job Recommendation System")

# User Inputs
job_title = st.text_input("Enter Job Title:")
work_type = st.selectbox("Select Work Type:", ["", "Full-Time", "Part-Time", "Remote", "Intern"])
preference = st.selectbox("Select Preference:", ["", "Male", "Female", "Both"])

# Recommend Jobs function
def recommend_jobs(job_title, work_type, preference):
    filtered_jobs = job_df_sample.copy()
    
    if work_type:
        filtered_jobs = filtered_jobs[filtered_jobs['Work Type'].str.contains(work_type, case=False, na=False)]
    if preference:
        filtered_jobs = filtered_jobs[filtered_jobs['Preference'].str.contains(preference, case=False, na=False)]
    if filtered_jobs.empty:
        return None, "No jobs found matching the given criteria."
    
    filtered_jobs = filtered_jobs.sort_values(by='Job Posting Date', ascending=False)
    
    try:
        indx = filtered_jobs[filtered_jobs['Job Title'].str.contains(job_title, case=False, na=False)].index[0]
        indx = job_df_sample.index.get_loc(indx)
    except IndexError:
        return None, "The specified job title does not exist in the dataset."
    
    # Use session state similarity matrix
    similarity = st.session_state.similarity
    distances = sorted(list(enumerate(similarity[indx])), key=lambda x: x[1], reverse=True)[1:20]
    
    jobs = [job_df_sample.iloc[i[0]] for i in distances if job_df_sample.iloc[i[0]].name in filtered_jobs.index]
    if not jobs:
        return None, "No similar jobs found within the filtered criteria."
    
    recommended_df = pd.DataFrame(jobs)
    display_cols = ['Job Title', 'Work Type', 'Preference', 'Experience', 'Qualifications', 
                    'Salary Range', 'location', 'Country', 'Contact Person', 'Contact', 
                    'Role', 'Job Portal', 'Job Description', 'Benefits', 'skills', 
                    'Responsibilities', 'Company', 'Job Posting Date']
    recommended_df = recommended_df[display_cols].sort_values(by='Job Posting Date', ascending=False)
    
    return recommended_df, None

# Display recommendations
if st.button("Find Jobs"):
    if not job_title and not work_type and not preference:
        st.warning("Please provide at least one input to filter jobs.")
    else:
        recommendations, error_message = recommend_jobs(job_title, work_type, preference)
        if error_message:
            st.warning(error_message)
        else:
            st.success("Recommended Jobs (sorted by most recent posting):")
            st.dataframe(recommendations)
