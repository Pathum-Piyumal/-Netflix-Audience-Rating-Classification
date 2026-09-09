import os
import sys

# Ensure project root directory is in Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.predict import predict_rating

st.set_page_config(
    page_title="Netflix Rating Classifier",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Rating Classification")
st.markdown("""
Predict audience rating category (**Adults**, **Teens**, or **Kids/Family**) for any Netflix title using Machine Learning.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Title Metadata Input")
    content_type = st.selectbox("Content Type", ["Movie", "TV Show"])
    release_year = st.number_input("Release Year", min_value=1940, max_value=2026, value=2021)
    duration = st.number_input("Duration (Minutes for Movies / Seasons for TV)", min_value=1, max_value=500, value=90)
    country = st.text_input("Country of Origin", value="United States")
    genres = st.text_input("Genres / Listed In", value="Children & Family Movies, Comedies")

with col2:
    st.subheader("Live Prediction")
    if st.button("Classify Content Rating", type="primary"):
        input_payload = {
            'type': content_type,
            'release_year': release_year,
            'duration_num': float(duration),
            'listed_in': genres,
            'country': country
        }
        
        try:
            result = predict_rating(input_payload)
            st.success(f"**Predicted Category:** {result['predicted_category']}")
            
            st.markdown("#### Probability Breakdown:")
            prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Category', 'Probability'])
            st.bar_chart(prob_df.set_index('Category'))
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
