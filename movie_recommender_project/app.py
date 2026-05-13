import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="🎬 Pro Movie Discovery", layout="wide")

# Custom CSS for a dark, modern look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSelectbox div { cursor: pointer !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    # 1. Map target execution environments
    local_path = "ml-32m/movies.csv"
    parent_path = "../ml-32m/movies.csv"
    
    # PASTE YOUR DIRECT CLOUD STORAGE DOWNLOAD URL HERE:
    cloud_url = "https://your-storage-bucket-url.com/movies.csv"
    
    # 2. Resilient Path Traversal
    if os.path.exists(local_path):
        target_source = local_path
    elif os.path.exists(parent_path):
        target_source = parent_path
    else:
        # Fallback to direct cloud streaming when deployed on Streamlit Cloud
        target_source = cloud_url
        
    # 3. Stream data into memory
    df = pd.read_csv(target_source)
    
    # ... keep your existing vote_count and TF-IDF metadata processing logic here ...
    df['metadata'] = df['title'] + " " + df['genres'].str.replace('|', ' ')
    return df

@st.cache_resource
def build_engine(df):
    # We use a wider ngram range (1,2) to capture movie series like "Toy Story"
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    return tfidf_matrix

def get_pro_recommendations(title, df, tfidf_matrix):
    try:
        idx = df.index[df['title'] == title].tolist()[0]
        # Compute similarity for this movie against all others
        # Using linear_kernel is memory efficient for large datasets
        cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Create a temp dataframe for ranking
        temp_df = df.copy()
        temp_df['sim_score'] = cosine_sim
        
        # --- THE SECRET SAUCE: HYBRID RANKING ---
        # We multiply similarity by a log of vote_count to favor popular movies
        # This stops obscure 1975 films from appearing unless they are VERY similar
        import numpy as np
        temp_df['final_score'] = temp_df['sim_score'] * (np.log1p(temp_df['vote_count']) + 1)
        
        # Sort and return top 10 (excluding itself)
        recommendations = temp_df[temp_df['title'] != title].sort_values(by='final_score', ascending=False)
        return recommendations.head(10)
    except:
        return None

# --- MAIN UI ---
df = load_and_prep_data()
tfidf_matrix = build_engine(df)

st.title("🎬 AI Movie Discovery Engine")
st.caption("Hybrid Engine: TF-IDF + Popularity Weighted Ranking")

selected_movie = st.selectbox("Search a movie you love:", ["-- Select --"] + list(df['title'].values))

if selected_movie != "-- Select --":
    results = get_pro_recommendations(selected_movie, df, tfidf_matrix)
    
    if results is not None:
        cols = st.columns(3)
        for i, (index, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"#### {row['title']}")
                    st.caption(f"🎭 {row['genres'].replace('|', ' • ')}")
                    
                    # Simulated match logic
                    match = 99 - (i * 1.5)
                    st.write(f"**Relevance:** {match:.1f}%")
                    st.progress(match / 100)
                    st.divider()
    else:
        st.error("Model couldn't find a match for that title.")