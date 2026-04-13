import streamlit as st
import pandas as pd
import re
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Resume Screening System", layout="wide", page_icon="🤖")

# ── Login credentials ─────────────────────────────────────────────────────────
USERS = {
    "admin": "1234",
    "hr":    "hr123"
}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ── Login page CSS ────────────────────────────────────────────────────────────
LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0d0f14 !important;
    color: #e8eaf0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 480px !important; margin: auto !important; }
[data-testid="stTextInput"] input {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 0.6rem 0.9rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6C8EF5 !important;
    box-shadow: 0 0 0 1px rgba(108,142,245,0.3) !important;
}
[data-testid="stTextInput"] label {
    color: #8890a8 !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6C8EF5, #8B6EF5) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stSuccess"] {
    background: rgba(78,205,196,0.08) !important;
    border: 1px solid rgba(78,205,196,0.2) !important;
    border-radius: 10px !important; color: #4ECDC4 !important;
}
[data-testid="stAlert"] {
    background: rgba(255,107,107,0.08) !important;
    border: 1px solid rgba(255,107,107,0.2) !important;
    border-radius: 10px !important;
}
</style>
"""

# ── Login page ────────────────────────────────────────────────────────────────
def login_page():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-bottom:2.5rem;padding-top:3rem;">
        <div style="width:56px;height:56px;border-radius:16px;
            background:linear-gradient(135deg,#6C8EF5,#4ECDC4);
            display:flex;align-items:center;justify-content:center;
            font-size:26px;margin:0 auto 1rem;">🤖</div>
        <div style="font-size:22px;font-weight:600;letter-spacing:-0.02em;">AI Resume Screening system</div>
        <div style="font-size:12px;color:#8890a8;margin-top:4px;">Sign in to continue</div>
    </div>
    <div style="background:#161820;border:1px solid rgba(255,255,255,0.07);
        border-radius:16px;padding:2rem;">
    """, unsafe_allow_html=True)

    username = st.text_input("USERNAME")
    password = st.text_input("PASSWORD", type="password")
    st.info("Demo Login → admin / 1234")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Sign In"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
     login_page()
     st.stop()


# ── Main app CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0d0f14 !important;
    color: #e8eaf0 !important;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: #161820 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8890a8 !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6C8EF5 !important;
    background: rgba(108,142,245,0.05) !important;
}
[data-testid="stFileUploader"] span { color: #8890a8 !important; font-size: 12px !important; }
[data-testid="stFileUploader"] button {
    background: rgba(108,142,245,0.15) !important; color: #6C8EF5 !important;
    border: 1px solid rgba(108,142,245,0.25) !important;
    border-radius: 7px !important; font-size: 12px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #1e2130 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
}
[data-testid="stSelectbox"] svg { fill: #8890a8 !important; }

[data-testid="stTextArea"] textarea {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; line-height: 1.65 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6C8EF5 !important;
    box-shadow: 0 0 0 1px rgba(108,142,245,0.3) !important;
}
[data-testid="stTextArea"] label {
    color: #8890a8 !important; font-size: 11px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6C8EF5, #8B6EF5) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important;
    letter-spacing: -0.01em !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stDownloadButton"] button {
    background: #161820 !important; color: #e8eaf0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    transition: border-color 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #6C8EF5 !important; color: #6C8EF5 !important;
}

[data-testid="stMetric"] {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8890a8 !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #6C8EF5 !important;
    font-family: 'DM Mono', monospace !important; font-size: 2rem !important;
}

[data-testid="stExpander"] {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.14) !important; }
[data-testid="stExpander"] summary {
    color: #e8eaf0 !important; font-weight: 500 !important; font-size: 14px !important;
}
[data-testid="stExpander"] summary:hover { color: #6C8EF5 !important; }

[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 4px !important; height: 6px !important;
}
[data-testid="stProgressBar"] > div > div { border-radius: 4px !important; }

[data-testid="stSuccess"] {
    background: rgba(78,205,196,0.08) !important;
    border: 1px solid rgba(78,205,196,0.2) !important;
    border-radius: 10px !important; color: #4ECDC4 !important;
}
[data-testid="stWarning"] {
    background: rgba(247,183,49,0.08) !important;
    border: 1px solid rgba(247,183,49,0.2) !important;
    border-radius: 10px !important;
}
[data-testid="stInfo"] {
    background: rgba(108,142,245,0.08) !important;
    border: 1px solid rgba(108,142,245,0.2) !important;
    border-radius: 10px !important; color: #6C8EF5 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; overflow: hidden !important;
}

hr { border-color: rgba(255,255,255,0.07) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Header banner ─────────────────────────────────────────────────────────────
username = st.session_state.get("username", "user")
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.5rem;">
    <div style="width:42px;height:42px;border-radius:12px;
        background:linear-gradient(135deg,#6C8EF5,#4ECDC4);
        display:flex;align-items:center;justify-content:center;
        font-size:20px;flex-shrink:0;">🤖</div>
    <div>
        <div style="font-size:20px;font-weight:600;letter-spacing:-0.02em;line-height:1.2;">AI Resume Screening System</div>
        <div style="font-size:12px;color:#8890a8;letter-spacing:0.01em;">Your AI hiring assistant</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
        <div style="font-size:12px;color:#8890a8;">
            Signed in as <span style="color:#6C8EF5;font-weight:600;">{username}</span>
        </div>
        <div style="background:rgba(78,205,196,0.1);color:#4ECDC4;
            border:1px solid rgba(78,205,196,0.2);border-radius:20px;
            padding:5px 14px;font-size:11px;font-weight:600;
            letter-spacing:0.05em;text-transform:uppercase;">● Ready</div>
    </div>
</div>
<hr style="margin:0.75rem 0 1.5rem;">
""", unsafe_allow_html=True)


# ── Core functions ────────────────────────────────────────────────────────────
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def extract_name(text):
    for line in text.split("\n")[:10]:
        line = line.strip()
        if re.match(r'^Name\s*[:\-]\s*(.+)', line, re.IGNORECASE):
            return re.match(r'^Name\s*[:\-]\s*(.+)', line, re.IGNORECASE).group(1).strip()
    for line in text.split("\n")[:5]:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and len(line) < 40 and line[0].isupper():
            return line
    return "Unknown"

def extract_email(text):
    match = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return match[0] if match else "Not Found"

def extract_phone(text):
    match = re.findall(r'\+?\d[\d\s\-]{8,}\d', text)
    return match[0].strip() if match else "Not Found"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

skills_list = [
    "python", "java", "sql", "machine learning", "data analysis", "nlp",
    "pandas", "numpy", "statistics", "data visualization",
    "recruitment", "hr", "communication", "management"
]

def extract_skills(text):
    text = text.lower()
    return [s for s in skills_list if s in text]

def missing_skills(resume, job_desc):
    job_skills = [s for s in skills_list if s in job_desc.lower()]
    resume_skills = extract_skills(resume)
    return [s for s in job_skills if s not in resume_skills]

def generate_pdf(results):
    doc = SimpleDocTemplate("results.pdf")
    styles = getSampleStyleSheet()
    content = [Paragraph("Resume Screening Report", styles['Title']), Spacer(1, 10)]
    for i, row in results.head(5).iterrows():
        text = (f"Candidate {i+1}<br/>"
                f"Name: {row['Name']}<br/>"
                f"Email: {row['Email']}<br/>"
                f"Phone: {row['Phone']}<br/>"
                f"Match: {row['Match %']}%<br/>"
                f"Skills: {row['Skill_Count']}<br/><br/>")
        content.append(Paragraph(text, styles['Normal']))
        content.append(Spacer(1, 10))
    doc.build(content)
    with open("results.pdf", "rb") as f:
        return f.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload")

    csv_file = st.file_uploader("CSV Dataset", type=["csv"])
    st.markdown("---")

    pdf_files = st.file_uploader("PDF Resumes", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")

    # Default categories
    default_categories = [
        "Data Scientist",
        "Software Engineer",
        "Web Developer",
        "HR",
        "Data Analyst",
        "Machine Learning Engineer"
    ]

    # ✅ Category logic
    if csv_file:
        df = pd.read_csv(csv_file)
        categories = df["Category"].unique()
        selected_category = st.selectbox("🎯 Job Category", categories)
    else:
        selected_category = st.selectbox("🎯 Job Category", default_categories)

    st.markdown("---")

    
    run = st.button("🚀 Run Screening")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
job_description = st.text_area(
    "JOB DESCRIPTION",
    placeholder="Paste the job description here",
    height=120
)

if run:
    resumes = []

    if csv_file and selected_category:
        filtered_df = df[df["Category"] == selected_category]
        for _, row in filtered_df.iterrows():
            text = row["Resume"]
            resumes.append({
                "name":  extract_name(text),
                "text":  text,
                "email": extract_email(text),
                "phone": extract_phone(text)
            })

    if pdf_files:
        for pdf in pdf_files:
            text = extract_pdf_text(pdf)
            resumes.append({
                "name":  extract_name(text),
                "text":  text,
                "email": extract_email(text),
                "phone": extract_phone(text)
            })

    if resumes and job_description:
        with st.spinner("Vectorising resumes and computing similarity…"):
            clean_resumes = [clean_text(r["text"]) for r in resumes]
            clean_job     = clean_text(job_description)

            vectorizer   = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(clean_resumes + [clean_job])
            scores       = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

            results = pd.DataFrame({
                "Name":   [r["name"]  for r in resumes],
                "Email":  [r["email"] for r in resumes],
                "Phone":  [r["phone"] for r in resumes],
                "Resume": [r["text"]  for r in resumes],
                "Score":  scores.flatten()
            })

            results["Match %"]    = (results["Score"] * 100).round(2)
            results["Skill_Count"] = results["Resume"].apply(lambda x: len(extract_skills(x)))
            results = results.sort_values(by=["Score", "Skill_Count"], ascending=[False, False])

        # ── Stats ─────────────────────────────────────────────────────────────
        st.success("✅ Screening complete!")
        st.markdown("<br>", unsafe_allow_html=True)

        job_skills = [s for s in skills_list if s in job_description.lower()]
        top_match  = results.iloc[0]["Match %"] if len(results) else 0
        avg_match  = round(results["Match %"].mean(), 1) if len(results) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates",  len(results))
        c2.metric("Top Match",   f"{top_match}%")
        c3.metric("Avg Match",   f"{avg_match}%")
        c4.metric("Skills in JD", len(job_skills))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Candidate cards ────────────────────────────────────────────────────
        st.subheader("🏆 Top Candidates")

        rank_icons  = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for rank, (i, row) in enumerate(results.head(5).iterrows(), start=1):
            match_pct = float(row["Match %"])
            skills    = extract_skills(row["Resume"])
            miss      = missing_skills(row["Resume"], job_description)
            skill_cnt = int(row["Skill_Count"])

            with st.container(border=True):
                left, right = st.columns([5, 1])

                with left:
                    badge = "  🌟 **Best Match**" if rank == 1 else ""
                    st.markdown(f"### {rank_icons[rank-1]}  {row['Name']}{badge}")

                    # Contact info
                    st.markdown(
                        f"📧 `{row['Email']}`  &nbsp;|&nbsp;  📞 `{row['Phone']}`",
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                    if skills:
                        st.markdown("**Matched skills:** " + "  ".join(f"`{s}`" for s in skills))
                    else:
                        st.markdown("_No skills matched_")

                    if miss:
                        st.markdown("**Missing skills:** " + "  ".join(f"`{s}`" for s in miss))
                    else:
                        st.markdown("✅ **All required skills matched**")

                with right:
                    st.metric(label="Match", value=f"{match_pct}%")
                    st.caption(f"{skill_cnt} skills found")

                st.progress(min(match_pct / 100.0, 1.0))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Downloads ──────────────────────────────────────────────────────────
        d1, d2 = st.columns(2)
        with d1:
            csv_data = results.drop(columns=["Resume"]).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV Report", csv_data, "results.csv", use_container_width=True)
        with d2:
            pdf_data = generate_pdf(results)
            st.download_button("📄 Download PDF Report", pdf_data, "results.pdf",
                               mime="application/pdf", use_container_width=True)
    else:
        st.warning("Please upload resumes and enter a job description.")