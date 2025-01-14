import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

#  CNN model
class IrisCNN(nn.Module):
    def __init__(self):
        super(IrisCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(128* 2, 64)  
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension (batch_size, 1, N_features)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flattening the tensor to pass it into the fully connected layer
        x = x.view(x.size(0), -1)  
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = IrisCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    epochs =20
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        output = model(X_train_tensor)

        loss = criterion(output, y_train_tensor)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        train_accuracy = (predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)

        train_losses.append(loss.item())
        test_accuracies.append(train_accuracy * 100)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%')

    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        _, predicted = torch.max(output, 1)
        test_accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')

    torch.save(model.state_dict(), 'iris_cnn_model2.pth')
    print("Model saved as 'iris_cnn_model2.pth'")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), test_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
