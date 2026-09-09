# Netflix Rating Classification

This repository implements a machine learning classification pipeline to categorize Netflix titles into target audience rating groups (**Adults**, **Teens**, or **Kids/Family**).

---

## 📊 Pipeline Results & Evaluation Summary

| Target Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Adults** | 0.61 | 0.89 | **0.73** | 802 |
| **Kids/Family** | **0.92** | 0.56 | **0.70** | 367 |
| **Teens** | 0.59 | 0.35 | 0.44 | 529 |
| **Overall Accuracy** | | | **65.08%** | 1698 |

![Confusion Matrix](reports/figures/confusion_matrix.png)

---

## 📁 Repository Structure

```
netflix-rating-classification/
│
├── data/
│   ├── raw/
│   │   └── netflix_titles.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Preprocessing & ColumnTransformer logic
│   ├── train.py              # Stratified CV & Hyperparameter tuning
│   ├── evaluate.py           # Metrics computation & plot generation
│   └── predict.py            # Live inference function
│
├── models/
│   └── best_rating_model.joblib
│
├── reports/
│   ├── figures/
│   │   └── confusion_matrix.png
│   └── results/
│       └── evaluation_metrics.json
│
├── app/
│   └── app.py                # Streamlit web application interface
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py                   # Master execution entry point
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model & Run Evaluation Pipeline
```bash
python main.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app/app.py
```
