# phase5_recommender.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load dataset (study_tips.csv created in Phase 4)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "data", "study_tips.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)
# Ensure columns: 'subject','text','study_tips'
print("Loaded dataset rows:", len(df))

# 2) Build TF-IDF matrix from the 'text' column
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'].astype(str))

# 3) helper function: recommend top N related subjects for a given subject name
def recommend_by_subject(subject_name, top_n=3):
    # find index for requested subject (first match)
    matches = df[df['subject'].str.lower() == subject_name.lower()]
    if matches.empty:
        print(f"No exact match for subject '{subject_name}'. Try partial or different name.")
        return []
    idx = matches.index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # exclude itself
    sims[idx] = -1
    top_idx = sims.argsort()[-top_n:][::-1]
    results = []
    for i in top_idx:
        results.append({
            "subject": df.at[i, 'subject'],
            "similarity": float(sims[i]),
            "study_tips": df.at[i, 'study_tips'],
            "text": df.at[i, 'text']
        })
    return results

# 4) helper function: recommend by free-text (user enters a short text)
def recommend_by_text(query_text, top_n=3):
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    results = []
    for i in top_idx:
        results.append({
            "subject": df.at[i, 'subject'],
            "similarity": float(sims[i]),
            "study_tips": df.at[i, 'study_tips'],
            "text": df.at[i, 'text']
        })
    return results

# 5) Small interactive CLI test
if __name__ == "__main__":
    print("\nAI Study Pal — TF-IDF Recommender\n")
    print("Subjects in dataset:", list(df['subject'].unique()))
    while True:
        mode = input("\nChoose mode: (1) by subject name, (2) by text, (q) quit: ").strip()
        if mode == 'q':
            break
        if mode == '1':
            s = input("Enter subject (exact match recommended): ").strip()
            recs = recommend_by_subject(s, top_n=3)
            if not recs:
                continue
            print(f"\nTop recommendations for '{s}':")
            for r in recs:
                print(f"- {r['subject']} (score: {r['similarity']:.3f})")
                print(f"  Tip: {r['study_tips']}")
        elif mode == '2':
            txt = input("Enter a short text describing what you want to study: ").strip()
            recs = recommend_by_text(txt, top_n=3)
            print(f"\nTop recommendations for your text:")
            for r in recs:
                print(f"- {r['subject']} (score: {r['similarity']:.3f})")
                print(f"  Tip: {r['study_tips']}")
        else:
            print("Invalid option.")
