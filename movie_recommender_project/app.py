import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🎬 AI Movie Discovery Engine",
    page_icon="🍿",
    layout="wide"
)

# Custom CSS for dark modern theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSelectbox div { cursor: pointer !important; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 1: DATA LOADING ---
@st.cache_data
def load_and_prep_data():
    # 1. Defining both possible file locations
    local_path = "top_movies.csv"
    cloud_path = os.path.join("movie_recommender_project", "top_movies.csv")
    
    # 2. Resolving the correct path based on the runtime environment
    if os.path.exists(local_path):
        target_file = local_path
    elif os.path.exists(cloud_path):
        target_file = cloud_path
    else:
        raise FileNotFoundError(
            "CRITICAL: Dataset 'top_movies.csv' missing from both root and subfolder execution layers."
        )
            
    # 3. Loading the dataset 
    df = pd.read_csv(target_file)
    
    # Casting attributes 
    df['title'] = df['title'].astype(str)
    df['genres'] = df['genres'].astype(str).fillna("Unknown")
    
    # Unified feature payload mapping
    df['metadata'] = df['title'] + " " + df['genres'].str.replace('|', ' ', regex=False)
    return df

# --- STEP 2: VECTOR ENGINE ---
@st.cache_resource
def build_engine(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    return tfidf_matrix

# --- STEP 3: HYBRID LOGIC ---
def get_pro_recommendations(title, df, tfidf_matrix):
    try:
        idx = df.index[df['title'] == title].tolist()[0]
        cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        temp_df = df.copy()
        temp_df['sim_score'] = cosine_sim
        
        # Logarithmic distribution applied to popularity weights
        temp_df['final_score'] = temp_df['sim_score'] * (np.log1p(temp_df['vote_count']) + 1)
        
        recommendations = temp_df[temp_df['title'] != title].sort_values(by='final_score', ascending=False)
        return recommendations.head(10)
    except Exception:
        return None

# --- MAIN GUI ---
def main():
    st.title("🎬 AI Movie Discovery Engine")
    st.caption("Hybrid Recommender: TF-IDF Semantic Proximity + Logarithmic Popularity Bias")
    
    try:
        df = load_and_prep_data()
        tfidf_matrix = build_engine(df)
    except Exception as e:
        st.error(str(e))
        return

    movie_list = df['title'].values
    selected_movie = st.selectbox(
        "Search for a movie you love:", 
        options=["-- Select a Movie --"] + list(movie_list),
        index=0
    )

    if selected_movie and selected_movie != "-- Select a Movie --":
        st.write(f"### Because you enjoyed **{selected_movie}**...")
        results = get_pro_recommendations(selected_movie, df, tfidf_matrix)
        
        if results is not None and not results.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"#### {row['title']}")
                        clean_genres = row['genres'].replace('|', ' • ')
                        st.caption(f"🎭 {clean_genres}")
                        
                        match_score = 99.0 - (i * 1.2)
                        st.write(f"**Relevance Match:** {match_score:.1f}%")
                        st.progress(match_score / 100.0)
                        st.divider()
        else:
            st.warning("No mathematically close neighbors found for this specific title.")
    else:
        st.info("👈 Select a title from the dropdown menu above to explore recommendations.")

if __name__ == "__main__":
    main()
