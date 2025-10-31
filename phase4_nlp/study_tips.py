# =============================================
# AI Study Pal - Phase 4: NLP Study Tips (Fixed)
# =============================================

import os
import pandas as pd
from collections import Counter
import nltk

# 1️⃣ Set up local NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# 2️⃣ Download missing resources (punkt, punkt_tab, stopwords)
for pkg, subfolder in [('punkt', 'tokenizers'), ('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora')]:
    try:
        nltk.data.find(f"{subfolder}/{pkg}")
        print(f"✅ '{pkg}' already exists.")
    except LookupError:
        print(f"⬇️ Downloading '{pkg}'...")
        nltk.download(pkg, download_dir=nltk_data_dir)

# 3️⃣ Import tools after ensuring downloads
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 4️⃣ Sample data
data = {
    "subject": ["Math", "Science", "History", "English", "Computer Science"],
    "text": [
        "Mathematics improves problem-solving and analytical thinking.",
        "Science enhances curiosity and innovation through research.",
        "History teaches lessons from ancient and modern times.",
        "English helps express thoughts clearly through reading and writing.",
        "Computer Science deals with computation, algorithms, and programming."
    ]
}

df = pd.DataFrame(data)
print("\n🔹 Original Dataset:")
print(df)

# 5️⃣ Prepare stopwords
stop_words = set(stopwords.words('english'))

# 6️⃣ Generate study tips
study_tips = []

for text in df['text']:
    tokens = word_tokenize(text.lower())
    keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
    top_keywords = [word for word, _ in Counter(keywords).most_common(3)]
    study_tips.append(f"Focus on: {', '.join(top_keywords)} daily.")

df['study_tips'] = study_tips

print("\n🔹 Generated Study Tips:")
print(df)

# 7️⃣ Save output CSV to the main /data folder (next to sample_dataset.csv)
base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, "../data/study_tips.csv")

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)
print(f"\n✅ Study tips saved successfully to: {output_path}")

# 8️⃣ Show working directory (for debugging)
print("📂 Current working directory:", os.getcwd())
