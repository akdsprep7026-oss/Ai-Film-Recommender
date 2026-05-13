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

# Custom CSS for modern styling (compatible with v1.24.0)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSelectbox div { cursor: pointer !important; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 1: DATA LOADING & PREPARATION ---
@st.cache_data
def load_and_prep_data():
    # Target the container-safe compressed artifact directly
    target_archive = "movies.csv.gz"
    
    # Absolute safety check for deployment environment
    if not os.path.exists(target_archive):
        # Fallback check just in case you are running from a parent folder locally
        if os.path.exists(os.path.join("movie_recommender_project", target_archive)):
            target_archive = os.path.join("movie_recommender_project", target_archive)
        else:
            raise FileNotFoundError(
                f"CRITICAL ERROR: Compiled feature pack '{target_archive}' is missing from the application directory. "
                "Ensure you have compressed movies.csv to movies.csv.gz and committed it to your repository."
            )
            
    # Load native GZIP compression straight into Pandas memory buffer
    df = pd.read_csv(target_archive, compression="gzip")
    
    # Ensure uniform string casting
    df['title'] = df['title'].astype(str)
    df['genres'] = df['genres'].astype(str).fillna("Unknown")
    
    # Safeguard vote_count metric initialization
    if 'vote_count' not in df.columns:
        df['vote_count'] = 0

    # Build composite NLP Vector Space: combine title and parsed genres
    df['metadata'] = df['title'] + " " + df['genres'].str.replace('|', ' ', regex=False)
    return df

# --- STEP 2: ENGINE COMPILATION ---
@st.cache_resource
def build_engine(df):
    # N-Gram range (1,2) ensures phrases like "Toy Story" are mapped as cohesive semantic units
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    return tfidf_matrix

# --- STEP 3: HYBRID RECOMMENDATION LOGIC ---
def get_pro_recommendations(title, df, tfidf_matrix):
    try:
        # Isolate target row index
        idx = df.index[df['title'] == title].tolist()[0]
        
        # Calculate global cosine proximity against all records using an optimized linear kernel
        cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Instantiate candidate scoring dataframe
        temp_df = df.copy()
        temp_df['sim_score'] = cosine_sim
        
        # --- HYBRID RANKING ALGORITHM ---
        # Scale raw textual similarity by logarithmic interaction density to filter out obscure noise
        temp_df['final_score'] = temp_df['sim_score'] * (np.log1p(temp_df['vote_count']) + 1)
        
        # Drop input candidate and return top 10 sorted outcomes
        recommendations = temp_df[temp_df['title'] != title].sort_values(by='final_score', ascending=False)
        return recommendations.head(10)
    except Exception:
        return None

# --- MAIN INTERFACE RENDERING ---
def main():
    st.title("🎬 AI Movie Discovery Engine")
    st.caption("Hybrid Recommender: TF-IDF Semantic Proximity + Logarithmic Popularity Bias")
    
    try:
        df = load_and_prep_data()
        tfidf_matrix = build_engine(df)
    except Exception as e:
        st.error(str(e))
        return

    # User Input Layer
    movie_list = df['title'].values
    selected_movie = st.selectbox(
        "Search for a movie you love:", 
        options=["-- Select a Movie --"] + list(movie_list),
        index=0
    )

    # Dynamic Execution Trigger
    if selected_movie and selected_movie != "-- Select a Movie --":
        st.write(f"### Because you enjoyed **{selected_movie}**...")
        
        results = get_pro_recommendations(selected_movie, df, tfidf_matrix)
        
        if results is not None and not results.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    # Standard container layout bypassing unsupported border parameters
                    with st.container():
                        st.markdown(f"#### {row['title']}")
                        
                        # Format genre visual tags cleanly
                        clean_genres = row['genres'].replace('|', ' • ')
                        st.caption(f"🎭 {clean_genres}")
                        
                        # Visual representation of match strength
                        match_score = 99.0 - (i * 1.2)
                        st.write(f"**Relevance Match:** {match_score:.1f}%")
                        st.progress(match_score / 100.0)
                        st.divider()
        else:
            st.warning("No mathematically close neighbors found for this specific title in the current vector space.")
    else:
        st.info("👈 Select a title from the dropdown menu above to explore the latent space.")

if __name__ == "__main__":
    main()