import os
import sys
import pandas as pd
from src.train import train_evaluate_save
from src.predict import predict_rating

def main():
    print("=" * 60)
    print("      Netflix Rating Classification Pipeline Execution      ")
    print("=" * 60)
    
    raw_data_path = os.path.join('data', 'raw', 'netflix_titles.csv')
    model_save_path = os.path.join('models', 'best_rating_model.joblib')
    
    if not os.path.exists(raw_data_path):
        print(f"\n[Error] Raw data file missing: {raw_data_path}")
        return
        
    print(f"\n[1/3] Dataset found at {raw_data_path}.")
    print("[2/3] Executing hyperparameter tuning, cross-validation & model training...")
    
    train_evaluate_save(raw_data_path, model_save_path)
    
    print("\n[3/3] Running sample inference check...")
    sample_item = {
        'type': 'Movie',
        'release_year': 2021,
        'duration_num': 105.0,
        'description': 'An animated family adventure featuring magical creatures saving their hometown.',
        'listed_in': 'Children & Family Movies, Comedies',
        'country': 'United States'
    }
    
    inference_result = predict_rating(sample_item)
    print(f"\nSample Input Title: {sample_item['description'][:60]}...")
    print(f"Predicted Audience Rating Category: {inference_result['predicted_category']}")
    print(f"Probabilities: {inference_result['probabilities']}")
    
    print("\n" + "=" * 60)
    print(" Pipeline complete! Model saved to models/best_rating_model.joblib")
    print("=" * 60)

if __name__ == "__main__":
    main()
