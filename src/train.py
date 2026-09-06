import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from src.preprocessing import clean_data, build_preprocessor_pipeline, get_train_test_data
from src.evaluate import evaluate_and_save_results

def train_evaluate_save(data_path: str, model_save_path: str):
    """
    Loads raw dataset, cleans data, performs hyperparameter tuning with 5-fold Stratified CV,
    evaluates performance, saves confusion matrix/metrics, and exports trained model artifact.
    """
    print("Loading dataset...")
    df_raw = pd.read_csv(data_path)
    df_clean = clean_data(df_raw)
    
    X_train, X_test, y_train, y_test = get_train_test_data(df_clean)
    print(f"Data split ready. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    preprocessor = build_preprocessor_pipeline()
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold Stratified Cross-Validation & GridSearch tuning...")
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    print("\nEvaluating on unseen test set...")
    evaluate_and_save_results(best_pipeline, X_test, y_test)
    
    joblib.dump(best_pipeline, model_save_path)
    print(f"Best model artifact saved to {model_save_path}")

if __name__ == "__main__":
    train_evaluate_save('data/raw/netflix_titles.csv', 'models/best_rating_model.joblib')
