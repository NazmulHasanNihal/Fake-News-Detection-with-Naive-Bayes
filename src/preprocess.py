import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    data['title'] = data['title'].fillna('')
    data['text'] = data['text'].fillna('')
    data['content'] = data['title'] + ' ' + data['text']
    data['content'] = data['content'].apply(clean_text)
    data['label'] = data['label'].map({'FAKE': 1, 'REAL': 0})
    data[['content', 'label']].to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data('data/news_data.csv', 'data/processed_news.csv')
