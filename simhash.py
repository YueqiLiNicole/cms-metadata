
import pandas as pd
import numpy as np
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertTokenizer
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Loading and Preprocessing
# TODO: use the API to load it
def load_data(filepath, delimiter=';'):
    try:
        data = pd.read_csv(filepath, delimiter=delimiter)
        logging.info("Data loaded successfully")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def normalize_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    return ' '.join([token for token in tokens if token not in stop_words])

# Feature Extraction with TF-IDF
def get_tfidf_features(texts, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    logging.info("TF-IDF features generated successfully.")
    return tfidf_matrix, vectorizer.get_feature_names_out()

# BERT
def get_bert_embeddings(texts, model_name='bert-base-uncased', batch_size=32):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Set the model to inference mode
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_pooled_embeddings.numpy())

    final_embeddings = np.concatenate(embeddings, axis=0)
    logging.info("BERT embeddings generated successfully.")
    return final_embeddings

# SimHash Implementation
class EnhancedSimHash:
    def __init__(self, num_bits=128):
        self.num_bits = num_bits

    def compute(self, vector):
        threshold = np.median(vector)
        binary_vector = np.where(vector > threshold, 1, 0)
        fingerprint = 0
        for bit in binary_vector:
            fingerprint = (fingerprint << 1) | bit
        return fingerprint

def combine_features(tfidf_matrix, bert_embeddings):
    combined = np.hstack((tfidf_matrix.toarray(), bert_embeddings))
    simhash = EnhancedSimHash(num_bits=128)
    return np.array([simhash.compute(vec) for vec in combined])

# Step 4: Train and Evaluate a Machine Learning Model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info(f"Model trained and evaluated successfully with accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{report}")
    return accuracy, report

# Main Execution
if __name__ == "__main__":
    data = load_data('/path/to/your/data.csv')
    data['Normalized Field Label'] = data['Field Label'].apply(normalize_text)

    tfidf_matrix, feature_names = get_tfidf_features(data['Normalized Field Label'])
    bert_embeddings = get_bert_embeddings(data['Normalized Field Label'])

    combined_simhash = combine_features(tfidf_matrix, bert_embeddings)

    y = pd.factorize(data['Root'])[0]  # Encoding the 'Root' labels to numeric form
    accuracy, report = train_and_evaluate_model(combined_simhash, y)

    logging.info(f"Final Model Accuracy: {accuracy}")
    logging.info(f"Final Classification Report:\n{report}")