import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(data_path, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    data = pd.read_csv(data_path)
    X = data['content']
    y = data['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model('data/processed_news.csv')
