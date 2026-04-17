import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class RankingModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        if 'preference' not in self.data:
            raise ValueError("Data must contain a 'preference' column")

    def prepare_data(self):
        # Assuming 'preference' is the target and the rest are features
        X = self.data.drop('preference', axis=1)
        y = self.data['preference']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f'Accuracy: {accuracy}')
        print('Classification Report:\n', report)

    def predict(self, new_data):
        return self.model.predict(new_data)