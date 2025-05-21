import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

def train_model(data_path, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    data = pd.read_csv(data_path)
    X = data['content']
    y = data['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=5,
        ngram_range=(1, 2),
        max_features=15000
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LinearSVC(max_iter=2000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_model('data/processed_news.csv')
