import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def load_dataset():

    file_path = r"D:\MiniProject1\iris.data"  

   
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file not found. Please ensure it's downloaded and extracted.")

   
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(file_path, delimiter=',', header=None, names=column_names)

    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   

    return X, y

def visualize_random_samples(X, y, num_samples=5):
    """
    Visualizes random samples from the dataset as line plots.
    Args:
        X: Feature matrix (samples x features).
        y: Labels corresponding to each sample.
        num_samples: Number of random samples to visualize.
    """
    num_features = X.shape[1]
    samples = random.sample(range(len(X)), num_samples)

    plt.figure(figsize=(6, 4))
    for i, idx in enumerate(samples):
        plt.subplot(1, num_samples, i + 1)
        plt.plot(range(num_features), X[idx])
        plt.title(f"Label: {y[idx]}")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    X, y = load_dataset()
    visualize_random_samples(X, y, num_samples=5)
