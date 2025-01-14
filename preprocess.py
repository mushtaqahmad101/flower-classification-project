import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    
    file_path = r"D:\MiniProject1\iris.data"

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file not found. Please ensure it's downloaded and extracted.")

   
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(file_path, delimiter=',', header=None, names=column_names)

   
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print("Data preprocessing complete.")
