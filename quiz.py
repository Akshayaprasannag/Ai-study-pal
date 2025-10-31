# quiz.py
import random, re
from typing import List, Tuple, Optional
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Lightweight stopwords (no NLTK)
STOPWORDS = set("""
a an the and or but if while of to in on for from with at by as is are was were be been being
this that these those it its into over under about after before during between through up down out
do does did doing have has had having not no nor so too very can will just than then there here
you your yours he she they them his her their i me my we us our
""".split())

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())

def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def top_keywords(text: str, k=10) -> List[str]:
    toks = [w for w in tokenize_words(text) if w not in STOPWORDS and len(w) > 2]
    if not toks:
        return []
    from collections import Counter
    freq = Counter(toks)
    return [w for w, _ in freq.most_common(k)]

def choose_sentence_for_question(text: str) -> Optional[str]:
    sents = [s for s in split_sentences(text) if len(s.split()) >= 8]
    if not sents:
        return None
    return max(sents, key=lambda s: len(s))  # longest sentence

def make_cloze_question(sentence: str, doc_keywords: List[str]) -> Optional[Tuple[str, str, List[str]]]:
    toks = tokenize_words(sentence)
    cand = [w for w in toks if w not in STOPWORDS and len(w) > 3]
    if not cand:
        return None
    # Prefer keywords
    answer = next((w for w in cand if w in doc_keywords), cand[0])
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    stem = pattern.sub("_____", sentence, count=1)
    distractor_pool = [w for w in doc_keywords if w.lower() != answer.lower()][:10]
    return stem, answer, distractor_pool

def build_mcq(stem: str, answer: str, distractor_pool: List[str], k=3):
    distractors = []
    pool = [d for d in distractor_pool if d.lower() != answer.lower()]
    random.shuffle(pool)
    for d in pool:
        if len(distractors) < k and d not in distractors:
            distractors.append(d)
    while len(distractors) < k:
        suffix = random.choice(["ing", "s", "ed", "ion"])
        guess = (answer + suffix)[:12]
        if guess.lower() != answer.lower() and guess not in distractors:
            distractors.append(guess)
    options = distractors + [answer]
    random.shuffle(options)
    correct_index = options.index(answer)
    return {"question": stem, "options": options, "answer_idx": correct_index}

def pseudo_difficulty_label(q_text: str) -> int:
    # 0 = easy, 1 = medium
    length = len(q_text.split())
    uniq = len(set(tokenize_words(q_text)))
    return 1 if (length >= 12 or uniq >= 10) else 0  # slightly easier to get both classes

class QuizEngine:
    def __init__(self):
        self.cv = CountVectorizer(max_features=3000, ngram_range=(1,2))
        self.clf: Optional[LogisticRegression] = None
        self._use_heuristic = True

    def fit_difficulty_from_texts(self, texts: List[str]) -> Tuple[float, float]:
        # Create pseudo question stems (sentences) from texts
        qs = []
        for t in texts:
            for s in split_sentences(t):
                s = s.strip()
                if len(s.split()) >= 6:
                    qs.append(s)
        if not qs:
            # No data → heuristic fallback
            self._use_heuristic = True
            return 0.0, 0.0

        y = np.array([pseudo_difficulty_label(q) for q in qs])
        X = self.cv.fit_transform(qs)

        # If only one class, skip training and fallback to heuristic
        classes = np.unique(y)
        if len(classes) < 2:
            self._use_heuristic = True
            self.clf = None
            return 0.0, 0.0

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        self.clf = clf
        self._use_heuristic = False
        return accuracy_score(yte, yhat), f1_score(yte, yhat)

    def predict_difficulty(self, q_text: str) -> str:
        if self._use_heuristic or self.clf is None:
            return "medium" if pseudo_difficulty_label(q_text) == 1 else "easy"
        x = self.cv.transform([q_text])
        pred = self.clf.predict(x)[0]
        return "medium" if pred == 1 else "easy"

    def generate_quiz(self, texts: List[str], n_q=6) -> List[dict]:
        pool = []
        for t in texts:
            sent = choose_sentence_for_question(t)
            if not sent:
                continue
            kws = top_keywords(t, k=12)
            cloze = make_cloze_question(sent, kws)
            if not cloze:
                continue
            stem, ans, dpool = cloze
            mcq = build_mcq(stem, ans, dpool)
            mcq["difficulty"] = self.predict_difficulty(mcq["question"])
            pool.append(mcq)
        random.shuffle(pool)
        return pool[:n_q]

# ---- Simple resource suggestions per subject ----
RESOURCES = {
    "math": [
        ("Khan Academy - Algebra", "https://www.khanacademy.org/math/algebra"),
        ("Paul's Math Notes", "https://tutorial.math.lamar.edu/"),
        ("MIT OCW - Calculus", "https://ocw.mit.edu/courses/mathematics/")
    ],
    "science": [
        ("OpenStax Physics", "https://openstax.org/details/books/college-physics"),
        ("CrashCourse Biology", "https://www.youtube.com/playlist?list=PL3EED4C1D684D3ADF"),
        ("NASA Science", "https://science.nasa.gov/")
    ],
    "history": [
        ("CrashCourse History", "https://www.youtube.com/playlist?list=PL8dPuuaLjXtMczXZUmjb3m2_2tFQvxr4k"),
        ("BBC History", "https://www.bbc.co.uk/history"),
        ("Sourcebooks", "https://sourcebooks.fordham.edu/")
    ],
    "english": [
        ("Purdue OWL", "https://owl.purdue.edu/"),
        ("Grammarly Handbook", "https://www.grammarly.com/handbook/"),
        ("CommonLit", "https://www.commonlit.org/")
    ],
    "computer science": [
        ("CS50", "https://cs50.harvard.edu/x/"),
        ("Real Python", "https://realpython.com/"),
        ("Visual Algo", "https://visualgo.net/")
    ],
}

def suggest_resources(subject: str, k=3):
    items = RESOURCES.get(subject.lower(), [])
    return items[:k]