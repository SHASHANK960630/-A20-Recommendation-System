import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.metrics.pairwise import cosine_similarity




st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="centered"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        "netflix_titles.csv"
    )

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("")

    # Combine text
    df["combined_text"] = (
        df["description"]
        + " "
        + df["listed_in"]
    )

    # Lowercase
    df["clean_text"] = (
        df["combined_text"]
        .str.lower()
    )

    # Remove punctuation
    df["clean_text"] = df[
        "clean_text"
    ].apply(
        lambda text: re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )
    )

    
    df["clean_text"] = df[
        "clean_text"
    ].apply(
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

    df["clean_text"] = df[
        "clean_text"
    ].apply(
        remove_stopwords
    )

    return df



@st.cache_resource
def create_model(text):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        text
    )

    return vectorizer, matrix


df = load_data()

vectorizer, tfidf_matrix = create_model(
    df["clean_text"]
)



def recommend(item_name, top_n):

    matches = df[
        df["title"].str.lower()
        == item_name.lower()
    ]

    if matches.empty:
        return []

    index = matches.index[0]

    
    scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

   
    similar_indexes = scores.argsort()[
        ::-1
    ]

    recommendations = []

    for item_index in similar_indexes:

        if item_index == index:
            continue

        recommendations.append({
            "title":
                df.iloc[item_index]["title"],

            "type":
                df.iloc[item_index]["type"],

            "genre":
                df.iloc[item_index]["listed_in"],

            "similarity":
                round(
                    scores[item_index],
                    3
                )
        })

        if len(recommendations) >= top_n:
            break

    return recommendations




st.title(
    "🎬 Netflix Recommendation System"
)

st.write(
    "A content-based recommendation system "
    "using TF-IDF and cosine similarity."
)

st.divider()


st.subheader(
    "Select a Movie or TV Show"
)


# Movie/show dropdown
items = sorted(
    df["title"]
    .dropna()
    .unique()
    .tolist()
)


selected_item = st.selectbox(
    "Choose an item:",
    items
)



top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)



if st.button(
    "🎯 Get Recommendations"
):

    results = recommend(
        selected_item,
        top_n
    )

    st.subheader(
        f"Recommendations for: "
        f"{selected_item}"
    )

    if results:

        for i, result in enumerate(
            results,
            start=1
        ):

            st.write(
                f"### {i}. "
                f"{result['title']}"
            )

            st.write(
                f"**Type:** "
                f"{result['type']}"
            )

            st.write(
                f"**Genre:** "
                f"{result['genre']}"
            )

            st.write(
                f"**Similarity:** "
                f"{result['similarity']}"
            )

            st.divider()

    else:

        st.error(
            "No recommendations found."
        )


st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    f"Total Items: {len(df)}"
)

st.sidebar.write(
    f"TF-IDF Features: "
    f"{tfidf_matrix.shape[1]}"
)

st.sidebar.write(
    "Method: Content-Based Filtering"
)

st.sidebar.write(
    "Vectorization: TF-IDF"
)

st.sidebar.write(
    "Similarity: Cosine Similarity"
)