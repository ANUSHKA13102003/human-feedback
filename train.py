import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

class TextFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

class RankingModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def prepare_features(self, data):
        # Assuming 'text' and 'target' are column names in the dataset
        texts = data['text']
        targets = data['target']
        text_features = TextFeatureExtractor().fit_transform(texts)
        return text_features, targets

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        return mse, r2, mae

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)

    def save_metrics(self, metrics, file_path):
        with open(file_path, 'w') as f:
            json.dump(metrics, f)

if __name__ == '__main__':
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Check for raw_dataset.csv
    raw_data_path = 'data/raw_dataset.csv'
    if not os.path.exists(raw_data_path):
        # Placeholder for DatasetGenerator, which should generate a CSV if missing.
        print("Generating raw_dataset.csv...")
        # DatasetGenerator().generate(raw_data_path)  # Uncomment when DatasetGenerator is implemented

    # Loading data
    trainer = RankingModelTrainer()
    data = trainer.load_data(raw_data_path)
    X, y = trainer.prepare_features(data)

    # Train the model
    trainer.train(X, y)

    # Save the model and metrics
    trainer.save_model('models/ranking_model.pkl')
    metrics = trainer.evaluate(X, y)
    trainer.save_metrics({'mse': metrics[0], 'r2': metrics[1], 'mae': metrics[2]}, 'models/metrics.json')