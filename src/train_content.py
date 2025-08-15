# src/train_content.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load data
df = pd.read_csv("data/products.csv")

# Combine text fields for content
df["content"] = df["title"] + " " + df["category"] + " " + df["brand"] + " " + df["features_text"]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])

# Save artifacts
pickle.dump(df, open("data/products.pkl", "wb"))
pickle.dump(vectorizer, open("data/vectorizer.pkl", "wb"))
pickle.dump(tfidf_matrix, open("data/tfidf_matrix.pkl", "wb"))

print("Training complete! Vectorizer & matrix saved.")
