import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("train.csv")  # Kaggle se download karke rakho
df = df.fillna('')  # missing values ko handle karo


# Title + Text combine
df['content'] = df['title'] + " " + df['text']

# X = features, y = target
X = df['content']
y = df['label']  # 0 = Real, 1 = Fake


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("NB Accuracy:", accuracy_score(y_test, y_pred_nb))

def predict_news(text):
    vectorized = vectorizer.transform([text])
    result = model.predict(vectorized)
    return "Fake News" if result[0] == 1 else "Real News"

print(predict_news("PM launches a new scheme for students..."))
