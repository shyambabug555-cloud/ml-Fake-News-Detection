# Fake News Detection System

## Project Overview
This project is a **Machine Learning-based Fake News Detection System** that classifies news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** techniques.  
It is built with **Python** and uses **Logistic Regression** and **Naive Bayes** classifiers for prediction.  

## Features
- Classifies news text as **Real** or **Fake**  
- Uses **TF-IDF vectorization** for text preprocessing  
- Provides **accuracy metrics** and classification report  
- Easy to use: input a news article and get instant prediction  

## Dataset
- The model is trained on a sample **Fake News Dataset** (`train.csv`)  
- Columns: `id`, `title`, `author`, `text`, `label`  
- `label = 0` → Real News  
- `label = 1` → Fake News  

## Installation
1. Clone the repository:  
```bash
git clone <your-repo-link>


pip install pandas numpy scikit-learn nltk
python ml.py
predict_news("Your news text here")
