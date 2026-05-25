# AI Resume Analyzer

A clean Streamlit app that compares a resume PDF against a job description and returns a visual match report with ATS checks, skill gaps, and improvement suggestions.

## Live Demo

Try the live app: https://hamza-ra.streamlit.app

## Why This Project

Recruiters and applicant tracking systems scan resumes quickly. This app helps job seekers understand how well their resume matches a specific job post and what they can improve before applying.

## Features

- PDF resume upload
- Job description comparison
- Visual overall match score
- Score breakdown for keyword match, skills match, experience relevance, education match, and ATS compatibility
- Skills found and missing skills
- Matched keyword highlights
- Resume improvement suggestions
- Weak bullet point detection
- ATS checks for readable text, contact details, core sections, length, bullet usage, and long text blocks
- Downloadable HTML analysis report
- Clean sidebar-based interface with tabs and visual score bars

## Tech Stack

- Python
- Streamlit
- PyPDF2
- scikit-learn
- TF-IDF vectorization
- Cosine similarity

## Project Structure

```text
resume-analyzer/
|-- app.py
|-- requirements.txt
|-- Procfile
|-- README.md
`-- .gitignore
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/hamzafiazdev/ai-resume-analyzer-app.git
   cd ai-resume-analyzer-app
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal.

## How It Works

The app extracts text from the uploaded resume PDF, compares it with the pasted job description, and calculates a weighted score using TF-IDF similarity, skill matching, education signals, experience signals, and ATS compatibility checks.

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open `https://share.streamlit.io`.
3. Create a new app from this repository.
4. Set the main file path to `app.py`.
5. Deploy.

### Render

This repository includes a `Procfile` for Render:

```text
web: streamlit run app.py --server.headless true --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false
```

## Resume Value

This project demonstrates:

- NLP-style text similarity
- PDF text extraction
- Feature engineering and scoring logic
- Data-driven UI design
- Streamlit app development
- Deployment-ready Python project structure

## Future Improvements

- DOCX resume support
- PDF report export
- More advanced semantic matching with embeddings
- Role-based skill dictionaries
- Resume section extraction with more precise scoring

## Privacy Note

Uploaded resumes are processed in memory during the session. The app does not intentionally store resume files.
