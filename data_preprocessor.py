import pandas as pd

def load_data(file_path):
    """Loads data from the given file path."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Removes duplicates and handles missing values."""
    data = data.drop_duplicates()
    data = data.fillna(method='ffill')  # Forward fill to handle missing values
    return data

def preprocess_data(data):
    """Converts categorical variables to dummy variables."""
    data = pd.get_dummies(data)
    return data

def main():
    file_path = 'path/to/your/data.csv'  # Specify your data file path
    data = load_data(file_path)
    data = clean_data(data)
    data = preprocess_data(data)
    print(data.head())  # Display the first few rows of the preprocessed data

if __name__ == '__main__':
    main()