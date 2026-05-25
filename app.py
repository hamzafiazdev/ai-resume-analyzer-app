import html
import re
from collections import Counter

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "django",
    "flask",
    "streamlit",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "data visualization",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "git",
    "github",
    "rest api",
    "api",
    "html",
    "css",
    "tailwind",
    "bootstrap",
    "agile",
    "scrum",
    "communication",
    "leadership",
    "problem solving",
    "project management",
]

SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "objective"],
    "skills": ["skills", "technical skills", "core skills"],
    "experience": ["experience", "work experience", "employment", "projects"],
    "education": ["education", "academic", "qualification"],
    "contact": ["email", "phone", "linkedin", "github"],
}

EDUCATION_TERMS = [
    "bachelor",
    "master",
    "phd",
    "degree",
    "computer science",
    "software engineering",
    "data science",
    "information technology",
    "certification",
    "diploma",
]

ACTION_VERBS = [
    "built",
    "created",
    "developed",
    "designed",
    "implemented",
    "improved",
    "optimized",
    "automated",
    "analyzed",
    "launched",
    "managed",
    "led",
    "delivered",
    "reduced",
    "increased",
]

BULLET_PREFIXES = ("-", "*", "\u2022")


def extract_resume_text(uploaded_file):
    pdf = PdfReader(uploaded_file)
    pages = []

    for page in pdf.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip(), len(pdf.pages)


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_skills(text):
    normalized = normalize_text(text)
    return sorted({skill for skill in SKILLS if re.search(rf"\b{re.escape(skill)}\b", normalized)})


def keyword_match(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=60)
    matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(matrix)[0][1]

    terms = vectorizer.get_feature_names_out()
    resume_scores = matrix[0].toarray()[0]
    job_scores = matrix[1].toarray()[0]
    shared = [
        (term, min(resume_scores[index], job_scores[index]))
        for index, term in enumerate(terms)
        if resume_scores[index] > 0 and job_scores[index] > 0
    ]

    return similarity, [term for term, _ in sorted(shared, key=lambda item: item[1], reverse=True)[:15]]


def extract_top_keywords(text, limit=20):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b", text.lower())
    filtered_words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]
    return [word for word, _ in Counter(filtered_words).most_common(limit)]


def calculate_experience_relevance(resume_text, job_desc):
    resume_lower = normalize_text(resume_text)
    job_lower = normalize_text(job_desc)
    experience_terms = ["experience", "project", "built", "developed", "managed", "designed", "implemented"]
    overlap = sum(1 for term in experience_terms if term in resume_lower and term in job_lower)
    years = [int(match) for match in re.findall(r"(\d+)\+?\s*(?:years|yrs)", resume_lower)]
    years_score = min(max(years, default=0) / 5, 1)
    action_score = min(sum(1 for verb in ACTION_VERBS if verb in resume_lower) / 8, 1)

    return min((overlap / len(experience_terms) * 0.45) + (years_score * 0.25) + (action_score * 0.30), 1)


def calculate_education_match(resume_text, job_desc):
    resume_terms = {term for term in EDUCATION_TERMS if term in normalize_text(resume_text)}
    job_terms = {term for term in EDUCATION_TERMS if term in normalize_text(job_desc)}

    if not job_terms:
        return 0.75, sorted(resume_terms), []

    matched = resume_terms.intersection(job_terms)
    missing = job_terms.difference(resume_terms)
    return len(matched) / len(job_terms), sorted(matched), sorted(missing)


def detect_sections(text):
    normalized = normalize_text(text)
    found = {}

    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords)

    return found


def ats_checks(resume_text, page_count):
    sections = detect_sections(resume_text)
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    unreadable_ratio = 1 if not resume_text else sum(ch == "\ufffd" for ch in resume_text) / max(len(resume_text), 1)

    checks = []
    checks.append(("Readable text extraction", bool(resume_text) and unreadable_ratio < 0.01, "PDF text was extracted successfully."))
    checks.append(("Contact details", sections["contact"] or bool(re.search(r"\S+@\S+\.\S+", resume_text)), "Include email, phone, LinkedIn, or GitHub near the top."))
    checks.append(("Skills section", sections["skills"], "Add a clear Skills or Technical Skills section."))
    checks.append(("Experience or projects section", sections["experience"], "Add Experience, Work Experience, or Projects."))
    checks.append(("Education section", sections["education"], "Add a clear Education section."))
    checks.append(("Reasonable length", page_count <= 2, "Keep early-career resumes near 1 page and experienced resumes near 2 pages."))
    checks.append(("Bullet-friendly content", sum(line.startswith(BULLET_PREFIXES) for line in lines) >= 3, "Use concise bullet points for achievements."))
    checks.append(("No oversized text blocks", all(len(line) <= 180 for line in lines[:80]), "Break long paragraphs into scan-friendly bullets."))

    return checks


def weak_bullet_points(resume_text):
    candidates = [
        line.strip(" -*\t\u2022")
        for line in resume_text.splitlines()
        if line.strip().startswith(BULLET_PREFIXES)
    ]
    weak = []

    for bullet in candidates:
        lower_bullet = bullet.lower()
        has_metric = bool(re.search(r"\d+|%|\$|x\b", bullet))
        has_action = any(lower_bullet.startswith(verb) for verb in ACTION_VERBS)
        too_short = len(bullet.split()) < 8

        if too_short or not has_metric or not has_action:
            weak.append(bullet)

    return weak[:6]


def build_suggestions(missing_skills, missing_keywords, weak_bullets, education_missing):
    suggestions = []

    if missing_skills:
        suggestions.append(f"Add relevant missing skills if you can honestly support them: {', '.join(missing_skills[:8])}.")
    if missing_keywords:
        suggestions.append(f"Mirror important job-description language naturally: {', '.join(missing_keywords[:10])}.")
    if education_missing:
        suggestions.append(f"Clarify education or certifications related to: {', '.join(education_missing)}.")
    if weak_bullets:
        suggestions.append("Rewrite weak bullets with this pattern: action verb + task + tool/skill + measurable result.")
    suggestions.append("Prefer achievement-focused bullets such as 'Improved dashboard load time by 35% using optimized SQL queries.'")
    suggestions.append("Keep keywords human-readable; avoid stuffing the same term repeatedly.")

    return suggestions


def score_breakdown(keyword_score, skill_score, experience_score, education_score, ats_score):
    scores = {
        "Keyword match": keyword_score,
        "Skills match": skill_score,
        "Experience relevance": experience_score,
        "Education match": education_score,
        "ATS compatibility": ats_score,
    }
    weights = {
        "Keyword match": 0.30,
        "Skills match": 0.25,
        "Experience relevance": 0.20,
        "Education match": 0.10,
        "ATS compatibility": 0.15,
    }
    overall = sum(scores[item] * weights[item] for item in scores)

    return overall, scores


def percent(value):
    return round(value * 100, 1)


def render_html_report(overall, scores, found_skills, missing_skills, shared_keywords, suggestions, checks, weak_bullets):
    check_items = "".join(
        f"<li><strong>{html.escape(name)}:</strong> {'Pass' if passed else 'Needs work'} - {html.escape(note)}</li>"
        for name, passed, note in checks
    )
    suggestion_items = "".join(f"<li>{html.escape(item)}</li>" for item in suggestions)
    score_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{percent(score)}%</td></tr>"
        for label, score in scores.items()
    )
    weak_items = "".join(f"<li>{html.escape(item)}</li>" for item in weak_bullets) or "<li>No weak bullet points detected.</li>"

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Resume Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #172033; line-height: 1.5; margin: 40px; }}
    h1, h2 {{ color: #0f4c81; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 10px; text-align: left; }}
    .score {{ font-size: 32px; font-weight: 700; color: #0f766e; }}
  </style>
</head>
<body>
  <h1>Resume Analysis Report</h1>
  <p class="score">Overall Score: {percent(overall)}%</p>
  <h2>Score Breakdown</h2>
  <table><tbody>{score_rows}</tbody></table>
  <h2>Skills Found</h2>
  <p>{html.escape(', '.join(found_skills) or 'No tracked skills found.')}</p>
  <h2>Missing Skills</h2>
  <p>{html.escape(', '.join(missing_skills) or 'No major tracked skills missing.')}</p>
  <h2>Matched Keywords</h2>
  <p>{html.escape(', '.join(shared_keywords) or 'No strong keyword overlap found.')}</p>
  <h2>Weak Bullet Points</h2>
  <ul>{weak_items}</ul>
  <h2>ATS Compatibility</h2>
  <ul>{check_items}</ul>
  <h2>Improvement Suggestions</h2>
  <ul>{suggestion_items}</ul>
</body>
</html>
"""


def inject_styles():
    st.markdown(
        """
<style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    .app-kicker {
        color: #52616f;
        font-size: 0.98rem;
        margin-bottom: 1.25rem;
    }

    .score-wrap {
        align-items: center;
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        display: flex;
        gap: 1.25rem;
        padding: 1.25rem;
    }

    .score-ring {
        align-items: center;
        background: conic-gradient(#0f766e var(--score), #edf2f7 0);
        border-radius: 50%;
        display: flex;
        height: 156px;
        justify-content: center;
        min-width: 156px;
        width: 156px;
    }

    .score-ring-inner {
        align-items: center;
        background: #ffffff;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        height: 114px;
        justify-content: center;
        width: 114px;
    }

    .score-value {
        color: #102a43;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }

    .score-label {
        color: #66788a;
        font-size: 0.78rem;
        margin-top: 0.35rem;
        text-transform: uppercase;
    }

    .summary-number {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 1rem;
    }

    .summary-number strong {
        color: #102a43;
        display: block;
        font-size: 1.75rem;
        line-height: 1.1;
    }

    .summary-number span {
        color: #66788a;
        font-size: 0.86rem;
    }

    .bar-row {
        margin-bottom: 0.95rem;
    }

    .bar-head {
        display: flex;
        font-size: 0.9rem;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }

    .bar-track {
        background: #edf2f7;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }

    .bar-fill {
        background: #0f766e;
        border-radius: 999px;
        height: 10px;
    }

    .chip-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.65rem 0 0.25rem;
    }

    .chip {
        background: #eef7f6;
        border: 1px solid #cfe7e3;
        border-radius: 999px;
        color: #134e4a;
        font-size: 0.85rem;
        padding: 0.34rem 0.65rem;
    }

    .chip.missing {
        background: #fff7ed;
        border-color: #fed7aa;
        color: #9a3412;
    }

    .status-grid {
        display: grid;
        gap: 0.7rem;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }

    .status-tile {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 0.85rem;
    }

    .status-tile.pass {
        border-left: 4px solid #0f766e;
    }

    .status-tile.warn {
        border-left: 4px solid #c2410c;
    }

    .status-title {
        color: #102a43;
        font-weight: 650;
        margin-bottom: 0.2rem;
    }

    .status-note {
        color: #66788a;
        font-size: 0.84rem;
    }

    .action-list {
        display: grid;
        gap: 0.75rem;
    }

    .action-item {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        border-radius: 6px;
        color: #263445;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def render_score_ring(overall):
    score = percent(overall)
    st.markdown(
        f"""
<div class="score-wrap">
  <div class="score-ring" style="--score: {score}%;">
    <div class="score-ring-inner">
      <div class="score-value">{score}%</div>
      <div class="score-label">Overall</div>
    </div>
  </div>
  <div>
    <h3 style="margin: 0 0 0.4rem;">Resume match snapshot</h3>
    <p style="color: #52616f; margin: 0;">
      A quick visual summary of how well this resume aligns with the job description.
    </p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_summary_number(value, label):
    st.markdown(
        f"""
<div class="summary-number">
  <strong>{html.escape(str(value))}</strong>
  <span>{html.escape(label)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_score_bars(scores):
    rows = []
    for label, score in scores.items():
        value = percent(score)
        rows.append(
            f"""
<div class="bar-row">
  <div class="bar-head"><span>{html.escape(label)}</span><strong>{value}%</strong></div>
  <div class="bar-track"><div class="bar-fill" style="width: {value}%;"></div></div>
</div>
"""
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_chip_list(items, empty_text, missing=False, limit=18):
    if not items:
        st.caption(empty_text)
        return

    class_name = "chip missing" if missing else "chip"
    chips = "".join(f'<span class="{class_name}">{html.escape(item)}</span>' for item in items[:limit])
    st.markdown(f'<div class="chip-list">{chips}</div>', unsafe_allow_html=True)


def render_status_grid(checks):
    tiles = []
    for name, passed, note in checks:
        class_name = "pass" if passed else "warn"
        status = "Pass" if passed else "Review"
        tiles.append(
            f"""
<div class="status-tile {class_name}">
  <div class="status-title">{html.escape(name)}: {status}</div>
  <div class="status-note">{html.escape(note)}</div>
</div>
"""
        )
    st.markdown(f'<div class="status-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_action_cards(suggestions):
    cards = "".join(f'<div class="action-item">{html.escape(item)}</div>' for item in suggestions[:5])
    st.markdown(f'<div class="action-list">{cards}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
inject_styles()
st.title("AI Resume Analyzer")
st.markdown(
    '<div class="app-kicker">Clean resume-to-job analysis with visual scoring, ATS checks, and a downloadable report.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analyze")
    uploaded_file = st.file_uploader("Resume PDF", type="pdf")
    job_desc = st.text_area("Job description", height=260)
    st.caption("Your file is processed in memory for this session.")

if uploaded_file and job_desc.strip():
    try:
        resume_text, page_count = extract_resume_text(uploaded_file)

        if not resume_text:
            st.error("No readable text was found in this PDF. Try exporting the resume as a text-based PDF.")
            st.stop()

        resume_skills = find_skills(resume_text)
        job_skills = find_skills(job_desc)
        missing_skills = sorted(set(job_skills).difference(resume_skills))
        skill_score = len(set(resume_skills).intersection(job_skills)) / len(job_skills) if job_skills else 0.75

        keyword_score, shared_keywords = keyword_match(resume_text, job_desc)
        job_keywords = extract_top_keywords(job_desc)
        resume_keywords = set(extract_top_keywords(resume_text, limit=80))
        missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_keywords]

        experience_score = calculate_experience_relevance(resume_text, job_desc)
        education_score, education_matched, education_missing = calculate_education_match(resume_text, job_desc)
        checks = ats_checks(resume_text, page_count)
        ats_score = sum(1 for _, passed, _ in checks if passed) / len(checks)
        weak_bullets = weak_bullet_points(resume_text)
        suggestions = build_suggestions(missing_skills, missing_keywords, weak_bullets, education_missing)
        overall, scores = score_breakdown(keyword_score, skill_score, experience_score, education_score, ats_score)

        report = render_html_report(
            overall,
            scores,
            resume_skills,
            missing_skills,
            shared_keywords,
            suggestions,
            checks,
            weak_bullets,
        )

        hero_left, hero_right = st.columns([1.05, 1])
        with hero_left:
            render_score_ring(overall)
        with hero_right:
            top_metrics = st.columns(3)
            with top_metrics[0]:
                render_summary_number(len(resume_skills), "skills found")
            with top_metrics[1]:
                render_summary_number(len(missing_skills), "skill gaps")
            with top_metrics[2]:
                render_summary_number(f"{sum(1 for _, passed, _ in checks if passed)}/{len(checks)}", "ATS checks")

            st.write("")
            st.download_button(
                "Download HTML Report",
                data=report,
                file_name="resume-analysis-report.html",
                mime="text/html",
                use_container_width=True,
            )

        st.subheader("Score Breakdown")
        render_score_bars(scores)

        tab_overview, tab_skills, tab_ats, tab_suggestions = st.tabs(
            ["Overview", "Skills & Keywords", "ATS", "Suggestions"]
        )

        with tab_overview:
            overview_cols = st.columns(2)
            with overview_cols[0]:
                st.markdown("**Experience relevance**")
                st.progress(experience_score)
                st.caption("Based on project terms, years mentioned, and action-oriented wording.")
            with overview_cols[1]:
                st.markdown("**Education match**")
                st.progress(education_score)
                if education_missing:
                    st.caption(f"Review education terms: {', '.join(education_missing)}")
                else:
                    st.caption("Education signals look aligned or were not required in the job description.")

        with tab_skills:
            skill_cols = st.columns(2)
            with skill_cols[0]:
                st.markdown("**Skills found**")
                render_chip_list(resume_skills, "No tracked skills found.")
            with skill_cols[1]:
                st.markdown("**Missing skills**")
                render_chip_list(missing_skills, "No major tracked skills missing.", missing=True)

            st.markdown("**Matched keywords**")
            render_chip_list(shared_keywords, "No strong keyword overlap found.", limit=20)

        with tab_ats:
            render_status_grid(checks)

        with tab_suggestions:
            render_action_cards(suggestions)
            with st.expander("Weak bullet points", expanded=False):
                if weak_bullets:
                    for bullet in weak_bullets:
                        st.warning(bullet)
                else:
                    st.success("No weak bullet points detected.")

    except Exception as exc:
        st.error(f"Something went wrong while analyzing the resume: {exc}")
elif uploaded_file or job_desc.strip():
    st.info("Upload a resume and paste a job description to begin the analysis.")
else:
    st.info("Upload a resume and paste a job description from the sidebar to begin.")
