from flask import Flask, render_template, request, send_file
import os
import re
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from quiz import QuizEngine, suggest_resources  # quiz.py must be in the same folder

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CSV_TIPS = os.path.join(DATA_DIR, 'study_tips.csv')
SCHEDULE_CSV = os.path.join(DATA_DIR, 'schedule.csv')
QUIZ_CSV = os.path.join(DATA_DIR, 'quiz.csv')
USER_LOG = os.path.join(DATA_DIR, 'user_inputs.csv')

# Create a tiny sample tips CSV if missing (so app still runs)
if not os.path.exists(CSV_TIPS):
    sample = pd.DataFrame({
        "subject": ["Math","Science","History","English","Computer Science"],
        "text": [
            "Mathematics improves problem-solving and analytical thinking.",
            "Science explores the natural world through experiments and observations.",
            "History studies past events and civilizations.",
            "English focuses on language, grammar, and communication.",
            "Computer Science covers algorithms, computation, and programming."
        ],
        "study_tips": [
            "Focus on: problem, solving, analytical daily.",
            "Focus on: experiments, observation, curiosity daily.",
            "Focus on: events, timelines, causes daily.",
            "Focus on: grammar, writing, vocabulary daily.",
            "Focus on: algorithms, coding, logic daily."
        ]
    })
    sample.to_csv(CSV_TIPS, index=False)

# Load tips data
df = pd.read_csv(CSV_TIPS)[['subject','text','study_tips']].dropna()
SUBJECTS = sorted(df['subject'].unique().tolist())

# Build TF-IDF for recommender
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'].astype(str))

# Build Quiz Engine once (train difficulty classifier on available texts)
qe = QuizEngine()
acc, f1 = qe.fit_difficulty_from_texts(df['text'].astype(str).tolist())
QUIZ_ACC, QUIZ_F1 = acc, f1
print(f"[Quiz] Difficulty model — acc={acc:.2f}, f1={f1:.2f}")

# -------- K-Means Topic Clustering (MiniBatchKMeans supports sparse) --------
kmeans = None
cluster_keywords = {}
CLUSTER_ON = tfidf_matrix.shape[0] >= 2  # need at least 2 samples

if CLUSTER_ON:
    n_clusters = min(5, tfidf_matrix.shape[0])  # cannot exceed number of samples
    n_clusters = max(2, n_clusters)            # at least 2 clusters
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=42, batch_size=64)
    labels = kmeans.fit_predict(tfidf_matrix)
    df['cluster'] = labels

    # Build top keywords per cluster using centers
    terms = tfidf.get_feature_names_out()
    centers = kmeans.cluster_centers_
    for cid in range(n_clusters):
        center = centers[cid]
        top_idx = np.argsort(center)[-8:][::-1]
        words = [terms[i] for i in top_idx if center[i] > 0]
        cluster_keywords[cid] = ", ".join(words[:8])
else:
    df['cluster'] = -1

def cluster_info_for_subject(subject_name: str, max_peers=3):
    if not CLUSTER_ON:
        return None
    row = df[df['subject'].str.lower() == subject_name.lower()]
    if row.empty:
        return None
    idx = row.index[0]
    cid = int(row.iloc[0]['cluster'])
    peers = df[(df['cluster'] == cid) & (df.index != idx)]
    peers = peers[['subject', 'study_tips']].head(max_peers).to_dict(orient='records')
    return {
        "id": cid,
        "keywords": cluster_keywords.get(cid, ""),
        "peers": peers
    }

# -------- Study Plan --------
SCENARIO_TASKS = {
    'exam': ["Review notes", "Practice problems", "Revise mistakes", "Take mini-quiz"],
    'homework': ["Read assignment", "Outline answers", "Draft solutions", "Check work"]
}

def generate_study_plan(subject: str, total_hours: int, scenario: str = 'exam', days: int = 7, start_date: str = None):
    """
    Evenly distribute total_hours across 'days'. Ensures sum(hours) == total_hours.
    If some days get 0h, we add a rest/light review note.
    """
    if start_date:
        start = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    else:
        start = dt.date.today()

    base = total_hours // days
    remainder = total_hours % days
    tasks = SCENARIO_TASKS.get(scenario, SCENARIO_TASKS['exam'])

    rows = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        h = base + (1 if i < remainder else 0)
        if h <= 0:
            day_tasks = [f"{subject}: Rest or light review"]
        else:
            day_tasks = [f"{subject}: {t}" for t in tasks[:max(1, min(len(tasks), h))]]
        rows.append({"date": d.isoformat(), "subject": subject, "hours": h, "tasks": " | ".join(day_tasks)})
    return pd.DataFrame(rows)

# -------- Recommender --------
def recommend_by_subject(subject_name: str, top_n=3):
    matches = df[df['subject'].str.lower() == subject_name.lower()]
    if matches.empty:
        return recommend_by_text(subject_name, top_n)
    idx = matches.index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1  # exclude itself
    top_idx = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_idx].to_dict(orient='records')

def recommend_by_text(query_text: str, top_n=3):
    if not query_text.strip():
        return []
    q_vec = tfidf.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_idx].to_dict(orient='records')

def tips_for_subject(subject_name: str):
    row = df[df['subject'].str.lower() == subject_name.lower()]
    if row.empty:
        return ""
    return str(row.iloc[0]['study_tips'])

# -------- Simple Summarizer (no NLTK) --------
STOPWORDS = set("""
a an the and or but if while of to in on for from with at by as is are was were be been being
this that these those it its into over under about after before during between through up down out
do does did doing have has had having not no nor so too very can will just than then there here
you your yours he she they them his her their i me my we us our
""".split())

def tokenize_words(text: str):
    return re.findall(r"[A-Za-z]+", text.lower())

def simple_summary(text: str, target_words=50):
    if not text or not text.strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) == 1:
        return " ".join(text.split()[:target_words])

    from collections import Counter
    all_tokens = [w for w in tokenize_words(text) if w not in STOPWORDS]
    if not all_tokens:
        out, count = [], 0
        for s in sentences:
            w = len(s.split())
            if count + w <= target_words:
                out.append(s); count += w
        return " ".join(out)

    freq = Counter(all_tokens)
    scores = []
    for s in sentences:
        toks = [w for w in tokenize_words(s) if w not in STOPWORDS]
        score = sum(freq.get(w,0) for w in toks) + len(toks)*0.1
        scores.append((score, s))

    out, count = [], 0
    for _, s in sorted(scores, key=lambda x: x[0], reverse=True):
        w = len(s.split())
        if count + w <= target_words:
            out.append(s); count += w
        if count >= target_words:
            break
    return " ".join(out) if out else sentences[0]

# -------- Feedback --------
SUBJECT_TONES = {
    "math": "Great job tackling those problems",
    "science": "Awesome progress on concepts and experiments",
    "history": "Strong grasp of timelines and causes",
    "english": "Nice work on reading and writing skills",
    "computer science": "Solid coding and algorithm practice"
}

def motivational_feedback(subject: str, hours: int):
    base = SUBJECT_TONES.get(subject.lower(), "Nice progress")
    return f"{base}! Planning {hours}h this week is a solid move. Keep going!"

# -------- Routes --------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           subjects=SUBJECTS,
                           quiz_acc=QUIZ_ACC,
                           quiz_f1=QUIZ_F1)

@app.route('/generate', methods=['POST'])
def generate():
    subject = request.form.get('subject', SUBJECTS[0])
    hours = int(request.form.get('hours', '6'))
    scenario = request.form.get('scenario', 'exam')
    query = request.form.get('query', '')
    short_text = request.form.get('short_text', '')

    # Study plan + CSV
    plan_df = generate_study_plan(subject, hours, scenario=scenario, days=7)
    plan_df.to_csv(SCHEDULE_CSV, index=False)

    # Recommendations and tips
    recs = recommend_by_text(query, top_n=3) if query.strip() else recommend_by_subject(subject, top_n=3)
    subj_tips = tips_for_subject(subject)

    # Quiz from subject texts (fallback to all texts if not enough)
    texts_sub = df[df['subject'].str.lower() == subject.lower()]['text'].astype(str).tolist()
    if len(texts_sub) < 3:
        texts_sub = df['text'].astype(str).tolist()
    quiz = qe.generate_quiz(texts_sub, n_q=6)

    # Save quiz CSV if we generated any questions
    if quiz:
        quiz_df = pd.DataFrame([{
            "question": q["question"],
            "A": q["options"][0] if len(q["options"]) > 0 else "",
            "B": q["options"][1] if len(q["options"]) > 1 else "",
            "C": q["options"][2] if len(q["options"]) > 2 else "",
            "D": q["options"][3] if len(q["options"]) > 3 else "",
            "correct_index": q["answer_idx"],
            "difficulty": q["difficulty"],
        } for q in quiz])
        quiz_df.to_csv(QUIZ_CSV, index=False)

    # Resources
    resources = suggest_resources(subject, k=3)

    # Summary
    summary = simple_summary(short_text, target_words=50) if short_text.strip() else ""

    # Feedback
    feedback = motivational_feedback(subject, hours)

    # Cluster info
    cinfo = cluster_info_for_subject(subject)

    # Log user inputs for report
    pd.DataFrame([{
        "ts": dt.datetime.now().isoformat(timespec='seconds'),
        "subject": subject,
        "hours": hours,
        "scenario": scenario,
        "query": query,
        "short_text_len": len(short_text),
        "n_quiz": len(quiz)
    }]).to_csv(USER_LOG, mode='a', header=not os.path.exists(USER_LOG), index=False)

    return render_template('index.html',
                           subjects=SUBJECTS,
                           selected_subject=subject,
                           hours=hours,
                           scenario=scenario,
                           schedule=plan_df.to_dict(orient='records'),
                           recommendations=recs,
                           subj_tips=subj_tips,
                           query=query,
                           summary=summary,
                           feedback=feedback,
                           quiz=quiz,
                           resources=resources,
                           quiz_acc=QUIZ_ACC,
                           quiz_f1=QUIZ_F1,
                           cluster_info=cinfo,
                           csv_ready=os.path.exists(SCHEDULE_CSV))

@app.route('/download_schedule')
def download_schedule():
    if os.path.exists(SCHEDULE_CSV):
        return send_file(SCHEDULE_CSV, as_attachment=True)
    return "No schedule available yet. Generate one first.", 404

@app.route('/download_quiz')
def download_quiz():
    if os.path.exists(QUIZ_CSV):
        return send_file(QUIZ_CSV, as_attachment=True)
    return "No quiz available yet. Generate one first.", 404

if __name__ == '__main__':
    app.run(debug=True)