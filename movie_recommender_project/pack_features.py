import pandas as pd

# 1. Load your source dataset
df = pd.read_csv("movies.csv")

# 2. (Optional) Apply any standard pre-processing or feature trimming here to save RAM

# 3. Save directly to Gzip compression
df.to_csv("movies.csv.gz", index=False, compression="gzip")
print("Feature pack successfully compiled to movies.csv.gz")