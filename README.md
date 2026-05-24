# AI Resume Analyzer

This project is a Streamlit app that analyzes a resume PDF against a job description and calculates a match score.

## Features

- Upload a resume in PDF format
- Paste a job description
- View a similarity score between the resume and the job description

## Setup

1. Install the dependencies:

   ```bash
   pip install streamlit PyPDF2 scikit-learn
   ```

2. Run the app:

   ```bash
   streamlit run app.py
   ```

## Notes

- This app uses a simple text-based similarity approach powered by scikit-learn.
