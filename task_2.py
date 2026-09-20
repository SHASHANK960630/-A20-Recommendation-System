import pandas as pd
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


print("=" * 60)
print("TASK 2 - TEXT PREPROCESSING")
print("=" * 60)

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Handle missing values
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["listed_in"] = df["listed_in"].fillna("")

# Combine description and genre
df["combined_text"] = (
    df["description"] + " " + df["listed_in"]
)

print("\nOriginal Text:")
print(df["combined_text"].head())


# Convert to lowercase
df["clean_text"] = df["combined_text"].str.lower()


# Remove punctuation and special characters
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

    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


df["clean_text"] = df["clean_text"].apply(
    remove_stopwords
)


print("\nCleaned Text:")
print(
    df[
        ["title", "combined_text", "clean_text"]
    ].head(10)
)

print("\nTASK 2 COMPLETED")