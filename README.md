// Netflix Recommendation System

// Project Description

A content-based recommendation system built using Python,
Pandas, Scikit-learn and Streamlit.

The system recommends similar Netflix movies and TV shows
based on description and genre.

// Dataset

Netflix Movies and TV Shows dataset.

Source:

https://www.kaggle.com/datasets/shivamb/netflix-shows

// Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Git
- GitHub
- Render

// Method

The project uses content-based filtering.

Pipeline:

Dataset
→ Text Preprocessing
→ TF-IDF
→ Cosine Similarity
→ Recommendation
→ Streamlit

// Text Preprocessing

- Missing value handling
- Lowercase conversion
- Punctuation removal
- Extra whitespace removal
- Stopword removal

// TF-IDF

TF-IDF is used to convert text into numerical vectors.

// Cosine Similarity

Cosine similarity is used to calculate similarity
between Netflix titles.

// How to Run

Install dependencies:

```bash
pip install -r requirements.txt