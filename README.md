#Filmic: AI Film Recommendation Engine 
# 🎬 Production-Grade AI Movie Discovery Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B?logo=streamlit&logoColor=white)
![CUDA Ready](https://img.shields.io/badge/Hardware-RTX%204060%20Optimized-76B900?logo=nvidia&logoColor=white)

A comprehensive, hybrid machine learning project designed to solve large-scale retrieval problems using the **MovieLens 32M Dataset**. 

This repository features two highly optimized approaches to recommendation systems: a deep learning **Two-Tower Neural Network** engineered for latent space mapping, paired with a production-ready **Hybrid Content-Based Web Application** that solves extreme cold-start and global variance collapse via NLP TF-IDF vectorization and logarithmic popularity weighting.

---
Live Demo
https://ai-film-recommender-fdbywux2yg2yxku74xnihl.streamlit.app/

## 🧠 System Architecture

### 1. The Deep Learning Core (`train.py`)
Developed using `TensorFlow Recommenders (TFRS)`, this architecture builds separate continuous vector spaces for Users and Movies.
* **Balanced Item Tower:** Learns composite embedding geometries (128-dim Title Vector + 128-dim Genre Vector) mapped into an optimized $L_2$-normalized space.
* **Balanced User Tower:** Constructs high-capacity 256-dim embeddings representing deep historical user preference arrays.
* **Asynchronous Streaming Pipelines:** Implements `tf.data.Dataset` mapping with highly targeted prefetching (`tf.data.AUTOTUNE`) and batch processing (`batch_size=8192`) to maximize GPU memory throughput during training.
* **Vector Serialization:** Compiles standard candidate arrays natively into a high-speed `BruteForce` index outputted to an executable `.pb` signature graph.

### 2. The Semantic Hybrid Retrieval GUI (`app.py`)
To ensure bulletproof user interaction inside production frameworks, the presentation layer implements a **Hybrid NLP Engine**.
* **TF-IDF Vectorization:** Leverages `scikit-learn` to transform unified string configurations (Titles + Genres) into vectorized continuous spaces using an optimized multi-gram threshold (`ngram_range=(1,2)`).
* **Cosine Proximity Kernel:** Computes inner pairwise semantic distances via highly optimized, memory-efficient linear kernels.
* **Popularity Weighting Logic:** Solves standard cold-start hallucinations (e.g., returning obscure titles for popular blockbusters) by applying a target multiplier:
$$\text{Final Relevance} = \text{Cosine Similarity} \times (\ln(1 + \text{Vote Count}) + 1)$$

---

## 📂 Project Structure

```text
├── ml-32m/                  # Target storage volume for source databases
│   ├── movies.csv           # Movie metadata (Features: movieId, title, genres)
│   └── ratings.csv          # Interaction patterns (Features: userId, movieId, rating)
├── movie_model_index/       # Serialized TensorFlow serving signatures
├── app.py                   # Streamlit cloud-deployed interface
├── train.py                 # Core TFRS deep-learning model mapping script
├── requirements.txt         # Package execution dependencies
└── README.md                # Technical operational specification
