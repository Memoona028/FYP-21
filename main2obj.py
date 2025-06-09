# To run: python -m uvicorn main2obj:app --reload

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import date
import time
import google.generativeai as genai

# Paste your Gemini API key here
GOOGLE_API_KEY = "AIzaSyBf8xvjuwbpAc6Mgv9iJORrKbVYrgbxPbI"
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# ==============================================

class Experience(BaseModel):
    jobTitle: str
    companyName: str
    location: Optional[str]
    employmentType: str
    startDate: date
    endDate: Optional[date]
    description: Optional[str]

class Education(BaseModel):
    degree: str
    fieldOfStudy: str
    institution: str
    location: Optional[str]
    startDate: date
    endDate: Optional[date]
    description: Optional[str]

class SocialLinks(BaseModel):
    linkedin: Optional[HttpUrl]
    portfolio: Optional[HttpUrl]
    github: Optional[HttpUrl]
    dribbble: Optional[HttpUrl]

class JobPreferences(BaseModel):
    desiredTitle: str
    preferredLocations: Optional[List[str]]
    industries: Optional[List[str]]
    salaryExpectation: Optional[int]
    employmentTypes: Optional[List[str]]

# =======================
# Main Resume Request 

class ResumeRequest(BaseModel):
    firstName: str
    lastName: str
    phone: Optional[str]
    location: Optional[str]
    summary: str
    skills: List[str]
    education: List[Education]
    experience: List[Experience]
    projects: Optional[List[str]]
    certifications: Optional[List[str]]
    volunteer_work: Optional[List[str]]
    interests: Optional[List[str]]
    socialLinks: Optional[SocialLinks]
    jobPreferences: Optional[JobPreferences]

# =======================
# Load the Gemini model


model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# =======================
# Helper Functions


def generate_cv_content(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            return "No response generated."
        except Exception as e:
            print(f"Error generating content (Attempt {attempt + 1}): {e}")
            time.sleep(2)
    return "Error in generation after multiple attempts."

def format_list(items):
    return ", ".join(items) if items else ""

def format_education(education):
    return "\n".join(
        f"- {e.degree} in {e.fieldOfStudy} from {e.institution} ({e.startDate} to {e.endDate or 'Present'})"
        for e in education
    )

def format_experience(experience):
    return "\n".join(
        f"- {e.jobTitle} at {e.companyName}, {e.employmentType} ({e.startDate} to {e.endDate or 'Present'})"
        f"\n  {e.description or ''}"
        for e in experience
    )

# =======================
# CV Generation Logic
# =======================

def generate_cv(data: ResumeRequest):
    prompt = f"""
You are a resume writing assistant. Based on the following information, generate a professional, clean, Harvard-style, ATS-friendly resume. Use bold section titles and keep content concise. Ensure it fits within 1–2 pages. Avoid images or graphics.

Name: {data.firstName} {data.lastName}
Phone: {data.phone or 'N/A'}
Location: {data.location or 'N/A'}

Summary:
{data.summary}

Skills:
{format_list(data.skills)}

Education:
{format_education(data.education)}

Experience:
{format_experience(data.experience)}

Projects:
{format_list(data.projects)}

Certifications:
{format_list(data.certifications)}

Volunteer Work:
{format_list(data.volunteer_work)}

Interests:
{format_list(data.interests)}

Social Links:
{data.socialLinks.dict() if data.socialLinks else 'N/A'}

Job Preferences:
{data.jobPreferences.dict() if data.jobPreferences else 'N/A'}
    """

    print("Generating CV with Gemini...\n")
    cv_text = generate_cv_content(prompt)

    return {"cv": cv_text}

# =======================
# FastAPI Endpoint


@app.post("/generate-resume")
async def generate_resume(request: ResumeRequest):
    resume = generate_cv(request)
    return resume
