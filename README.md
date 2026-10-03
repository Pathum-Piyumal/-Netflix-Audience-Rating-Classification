# Netflix Content Rating Classification

This repository contains the machine learning pipeline to classify content ratings for Netflix titles.

## Folder Structure

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
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   ├── figures/
│   └── results/
│
├── app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```
