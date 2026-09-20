import pandas as pd

df = pd.read_csv("netflix_titles.csv")

print("Dataset Shape:")
print(df.shape)


print("\nColumn Names:")
print(df.columns.tolist())


print("\nFirst 5 Rows:")
print(df.head())


print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nText Columns:")
print(["title", "listed_in", "description"])
