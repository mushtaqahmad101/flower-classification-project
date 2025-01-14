import os
import requests
import pandas as pd

def download_dataset():
    # URL of the Iris dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = "iris.data"
    
    #
    if not os.path.exists(filename):
        print("Downloading dataset...")
        response = requests.get(url)
        print(f"Response content type: {response.headers.get('Content-Type')}")
        if response.status_code == 200:
            with open(filename, "wb") as file:
                file.write(response.content)
            print("Download complete!")
        else:
            print(f"Failed to download file. HTTP status code: {response.status_code}")
            return
    try:
        
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        df = pd.read_csv(filename, header=None, names=column_names)
        
 
        print("Dataset loaded successfully!")
        print(df.head())
    except Exception as e:
        print(f"Failed to read the dataset: {e}")

if __name__ == "__main__":
    download_dataset()
