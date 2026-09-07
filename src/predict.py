import os
import joblib
import pandas as pd
from typing import Dict, Any

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_rating_model.joblib')

def predict_rating(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts content rating category for a given title's metadata.
    
    Expected keys in input_data:
      - 'type': 'Movie' or 'TV Show'
      - 'release_year': int
      - 'duration_num': float
      - 'listed_in': str
      - 'country': str
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training script first.")
        
    model = joblib.load(MODEL_PATH)
    
    df_input = pd.DataFrame([input_data])
    
    prediction = model.predict(df_input)[0]
    probabilities = model.predict_proba(df_input)[0]
    classes = model.classes_
    
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    
    return {
        'predicted_category': prediction,
        'probabilities': prob_dict
    }

if __name__ == "__main__":
    sample_input = {
        'type': 'Movie',
        'release_year': 2022,
        'duration_num': 110.0,
        'listed_in': 'Action & Adventure, Dramas',
        'country': 'United States'
    }
    
    print("Sample Input:\n", sample_input)
    res = predict_rating(sample_input)
    print("\nPrediction Result:\n", res)
