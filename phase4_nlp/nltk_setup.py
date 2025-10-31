import nltk, os

nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)

print("🔍 Using nltk_data directory:", nltk_data_dir)

# Force re-download clean versions
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Check contents
print("✅ punkt path:", nltk.find('tokenizers/punkt'))
print("✅ stopwords path:", nltk.find('corpora/stopwords'))
