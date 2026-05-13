import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# ==========================================
# 1. SYSTEM INITIALIZATION & MAPPING
# ==========================================
# Note: Executing natively in high-precision Float32 space bypasses strict Keras 2 
# sub-graph auto-casting conflicts while preserving core CUDA core execution paths.
print("1. Mapping native execution targets...")

movies_path = os.path.join("ml-32m", "movies.csv")
ratings_path = os.path.join("ml-32m", "ratings.csv")

if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
    raise FileNotFoundError(
        "Missing target files. Ensure 'movies.csv' and 'ratings.csv' are extracted "
        "directly into the 'ml-32m' folder within your project directory."
    )

# ==========================================
# 2. DATA EXTRACTION & PIPELINE CONSTRUCTION
# ==========================================
print("2. Streaming target features into memory...")
# Prototyping payload set to 2.5 Million rows to ensure rapid verification cycles.
movies_df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
ratings_df = pd.read_csv(ratings_path, usecols=["userId", "movieId"], nrows=2500000)

print("3. Formatting metadata layers and isolating primary genres...")
merged_df = ratings_df.merge(movies_df, on="movieId", how="inner")

# Uniform String Formatting
merged_df["userId"] = merged_df["userId"].astype(str)
merged_df["title"] = merged_df["title"].astype(str)
merged_df["primary_genre"] = merged_df["genres"].apply(
    lambda x: x.split("|")[0] if isinstance(x, str) else "Unknown"
)

movies_df["title"] = movies_df["title"].astype(str)
movies_df["primary_genre"] = movies_df["genres"].apply(
    lambda x: x.split("|")[0] if isinstance(x, str) else "Unknown"
)

print("4. Compiling vocabulary indices...")
unique_user_ids = merged_df["userId"].unique()
unique_movie_titles = movies_df["title"].unique()

print("5. Packaging optimized asynchronous tf.data streaming inputs...")
ratings_dataset = tf.data.Dataset.from_tensor_slices({
    "user_id": merged_df["userId"].values,
    "movie_title": merged_df["title"].values,
    "movie_genre": merged_df["primary_genre"].values
})

unique_movies_df = movies_df.drop_duplicates(subset=["title"])
candidate_movies_dataset = tf.data.Dataset.from_tensor_slices({
    "movie_title": unique_movies_df["title"].values,
    "movie_genre": unique_movies_df["primary_genre"].values
})

# Complete explicit eviction of RAM allocations prior to graph initializations
del merged_df
del ratings_df
del movies_df
gc.collect()


# ==========================================
# 3. BALANCED TWO-TOWER ENGINE
# ==========================================

class BalancedItemTower(tf.keras.Model):
    def __init__(self, titles_vocab):
        super().__init__()
        self.title_lookup = tf.keras.layers.StringLookup(vocabulary=titles_vocab, mask_token=None)
        self.title_embedding = tf.keras.layers.Embedding(len(titles_vocab) + 1, 128)
        
        genre_vocab = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Unknown"
        ]
        self.genre_lookup = tf.keras.layers.StringLookup(vocabulary=genre_vocab, mask_token=None)
        self.genre_embedding = tf.keras.layers.Embedding(len(genre_vocab) + 1, 128)

    def call(self, inputs):
        title_emb = self.title_embedding(self.title_lookup(inputs["movie_title"]))
        genre_emb = self.genre_embedding(self.genre_lookup(inputs["movie_genre"]))
        
        # Concat outputs a natively stable precision geometry: (None, 256)
        combined_vector = tf.concat([title_emb, genre_emb], axis=1)
        return tf.math.l2_normalize(combined_vector, axis=-1)


class BalancedUserTower(tf.keras.Model):
    def __init__(self, users_vocab):
        super().__init__()
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=users_vocab, mask_token=None)
        self.user_embedding = tf.keras.layers.Embedding(len(users_vocab) + 1, 256)

    def call(self, inputs):
        if isinstance(inputs, dict):
            raw_embedding = self.user_embedding(self.user_lookup(inputs["user_id"]))
        else:
            raw_embedding = self.user_embedding(self.user_lookup(inputs))
            
        return tf.math.l2_normalize(raw_embedding, axis=-1)


class OptimizedRecommenderEngine(tfrs.Model):
    def __init__(self, users_vocab, titles_vocab, candidates_ds):
        super().__init__()
        self.user_model = BalancedUserTower(users_vocab)
        self.movie_model = BalancedItemTower(titles_vocab)
        
        # Native model mapping execution inside standard precision buffers
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=candidates_ds.batch(2048).map(self.movie_model)
        )
        self.task = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features)
        movie_embeddings = self.movie_model(features)
        return self.task(user_embeddings, movie_embeddings)


# ==========================================
# 4. COMPILATION & EXECUTION CONTROL
# ==========================================
print("6. Initializing Deep Learning Core Architecture...")
model = OptimizedRecommenderEngine(unique_user_ids, unique_movie_titles, candidate_movies_dataset)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))

print("7. Launching High-Speed Processing Passes (Target: 4 Epochs)...")
tf.random.set_seed(42)

train_dataset = (
    ratings_dataset
    .shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    .batch(8192)
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(train_dataset, epochs=4)


# ==========================================
# 5. PRODUCTION INDEX SERIALIZATION
# ==========================================
print("8. Packaging highly targeted vector search models...")
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

index.index_from_dataset(
    tf.data.Dataset.zip((
        candidate_movies_dataset.map(lambda x: x["movie_title"]).batch(2048),
        candidate_movies_dataset.batch(2048).map(model.movie_model)
    ))
)

print("9. Triggering variable mapping execution pass (Graph Warm-up)...")
_, _ = index(tf.constant(["42"]))

export_path = os.path.join(".", "movie_model_index")
tf.saved_model.save(index, export_path)

print(f"\nSUCCESS: Native engine training validated. Serving index saved directly to: '{export_path}'.")