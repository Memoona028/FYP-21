# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:26:25 2024

@author: DELL
"""

import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Ensure NLTK data is downloaded
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

# Load dataset and take a sample
@st.cache_data
def load_sampled_data():
    df = pd.read_csv("job_descriptions.csv")
    return df.sample(n=10000, random_state=42)  # Select a random sample of 10,000 rows

job_df = load_sampled_data()

# Preprocessing
ps = PorterStemmer()

def cleaning(txt):
    if pd.isnull(txt):  # Handle NaN values
        return ""
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)  # Remove special characters
    tokens = txt.lower().split()  # Tokenize by splitting on spaces
    stemming = [ps.stem(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(stemming)

@st.cache_data
def preprocess_sampled_data():
    for col in ['Job Title', 'Qualifications', 'location']:
        job_df[col] = job_df[col].astype(str).apply(cleaning)

preprocess_sampled_data()


# Streamlit App
st.title("Job Portal")

# User Profile Form
st.sidebar.header("User Profile")
if "profile" not in st.session_state:
    st.session_state["profile"] = {}

st.sidebar.write("Fill in your profile details:")

job_title = st.sidebar.text_input("Job Title", st.session_state["profile"].get("Job Title", ""))
qualification = st.sidebar.text_input("Qualification", st.session_state["profile"].get("Qualification", ""))
location = st.sidebar.text_input("Location", st.session_state["profile"].get("Location", ""))

if st.sidebar.button("Save Profile"):
    st.session_state["profile"] = {
        "Job Title": job_title,
        "Qualification": qualification,
        "Location": location,
    }
    st.sidebar.success("Profile saved!")

# Display User Profile on Main Page
st.subheader("Your Profile")
if st.session_state["profile"]:
    st.write(st.session_state["profile"])
else:
    st.info("No profile data entered. Please fill out your profile in the sidebar.")

# Filter Jobs Based on Profile
def filter_jobs(profile):
    filtered_jobs = job_df.copy()
    if "Job Title" in profile and profile["Job Title"]:
        filtered_jobs = filtered_jobs[filtered_jobs['Job Title'].str.contains(profile["Job Title"], case=False, na=False)]
    if "Qualification" in profile and profile["Qualification"]:
        filtered_jobs = filtered_jobs[filtered_jobs['Qualifications'].str.contains(profile["Qualification"], case=False, na=False)]
    if "Location" in profile and profile["Location"]:
        filtered_jobs = filtered_jobs[filtered_jobs['location'].str.contains(profile["Location"], case=False, na=False)]
    
    if filtered_jobs.empty:
        return None, "No jobs found matching your profile."
    
    return filtered_jobs, None

# Display Filtered Jobs
st.subheader("Jobs Matching Your Profile")
if st.session_state["profile"]:
    results, error_message = filter_jobs(st.session_state["profile"])
    if error_message:
        st.warning(error_message)
    else:
        display_cols = ['Job Title', 'Qualifications', 'location', 'Company', 'Salary Range', 'Job Description']
        st.dataframe(results[display_cols])
else:
    st.info("Please save your profile to see matching jobs.")
