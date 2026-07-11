import pandas as pd

# 1. Loading the source dataset
df = pd.read_csv("movies.csv")



# 3. Saving directly to Gzip compression
df.to_csv("movies.csv.gz", index=False, compression="gzip")
print("Feature pack successfully compiled to movies.csv.gz")
