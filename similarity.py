import pandas as pd
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.metrics.pairwise import cosine_similarity


print("=" * 60)
print("TASK 4 - COSINE SIMILARITY")
print("=" * 60)


# Load dataset
df = pd.read_csv("netflix_titles.csv")


# Handle missing values
df["description"] = df["description"].fillna("")
df["listed_in"] = df["listed_in"].fillna("")
df["title"] = df["title"].fillna("")


# Combine text
df["combined_text"] = (
    df["description"] + " " + df["listed_in"]
)


# Lowercase
df["clean_text"] = df["combined_text"].str.lower()


# Remove punctuation
df["clean_text"] = df["clean_text"].apply(
    lambda text: re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )
)


# Remove extra spaces
df["clean_text"] = df["clean_text"].apply(
    lambda text: re.sub(
        r"\s+",
        " ",
        text
    ).strip()
)


# Remove stopwords
def remove_stopwords(text):

    words = text.split()

    return " ".join(
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    )


df["clean_text"] = df["clean_text"].apply(
    remove_stopwords
)


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["clean_text"]
)


# ------------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------------

print("\nCalculating cosine similarity...")

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)


# Show similarity for first item
print("\nFirst Item:")
print(df.iloc[0]["title"])

print("\nSimilarity Scores:")
print(similarity_matrix[0][:10])


print("\nTASK 4 COMPLETED")