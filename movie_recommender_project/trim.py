import pandas as pd

print("1. Loading raw MovieLens datasets...")
# Load local files
movies = pd.read_csv("ml-32m/movies.csv")
# Load just enough ratings to extract accurate popularity weights
ratings = pd.read_csv("ml-32m/ratings.csv", usecols=['movieId'], nrows=5000000)

print("2. Calculating popularity weights...")
popularity = ratings['movieId'].value_counts().reset_index()
popularity.columns = ['movieId', 'vote_count']

print("3. Merging and isolating the top 25,000 primary titles...")
df = movies.merge(popularity, on='movieId', how='left').fillna(0)
# Sort by popularity and keep the top 25k blockbusters/cult classics
top_movies = df.sort_values(by='vote_count', ascending=False).head(25000)

# Save directly to your target deployment folder
top_movies.to_csv("movie_recommender_project/top_movies.csv", index=False)
print("SUCCESS: 'top_movies.csv' generated securely inside movie_recommender_project/")