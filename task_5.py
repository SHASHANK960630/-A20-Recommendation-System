import pandas as pd
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.metrics.pairwise import cosine_similarity


print("=" * 60)
print("TASK 5 - RECOMMENDATION SYSTEM")
print("=" * 60)




df = pd.read_csv("netflix_titles.csv")


# Missing values
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["listed_in"] = df["listed_in"].fillna("")


# Combine description + genre
df["combined_text"] = (
    df["description"] + " " + df["listed_in"]
)



df["clean_text"] = (
    df["combined_text"].str.lower()
)


df["clean_text"] = df["clean_text"].apply(
    lambda text: re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )
)


df["clean_text"] = df["clean_text"].apply(
    lambda text: re.sub(
        r"\s+",
        " ",
        text
    ).strip()
)


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



vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["clean_text"]
)




def recommend(item_name, top_n=5):


    matches = df[
        df["title"].str.lower()
        == item_name.lower()
    ]

    if matches.empty:

        print(
            f"\n'{item_name}' was not found."
        )

        return

    
    index = matches.index[0]

   
    similarity_scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    similar_indexes = similarity_scores.argsort()[
        ::-1
    ]

    print(
        f"\nRecommendations for: "
        f"{df.iloc[index]['title']}"
    )

    print("-" * 50)

    count = 0

    for item_index in similar_indexes:

       
        if item_index == index:
            continue

        title = df.iloc[item_index]["title"]
        genre = df.iloc[item_index]["listed_in"]
        item_type = df.iloc[item_index]["type"]
        score = similarity_scores[item_index]

        print(
            f"\n{count + 1}. {title}"
        )

        print(
            f"   Type: {item_type}"
        )

        print(
            f"   Genre: {genre}"
        )

        print(
            f"   Similarity: {score:.3f}"
        )

        count += 1

        if count == top_n:
            break



print("\nFirst 20 Available Titles:")

print(
    df["title"].head(20).to_string(
        index=False
    )
)



test_items = (
    df["title"]
    .dropna()
    .head(3)
    .tolist()
)


for item in test_items:

    recommend(
        item,
        top_n=5
    )


print("\n" + "=" * 60)
print("TASK 5 COMPLETED")
print("=" * 60)