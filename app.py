from flask import Flask, request, jsonify, render_template
import torch
from train_model import IrisCNN  
import numpy as np
import os


app = Flask(__name__)
model_path = "iris_cnn_model2.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

# Load the model
input_size = 4 
num_classes = 3  
model = IrisCNN() 
model.load_state_dict(torch.load(model_path))  # Load the model weights
model.eval()  

class_names = ["Setosa", "Versicolor", "Virginica"]
@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    if "features" not in data:
        return jsonify({"error": "No 'features' key found in the request data"}), 400

    try:
        input_data = np.array(data["features"]).reshape(1, -1)
        if input_data.shape[1] != input_size:
            return jsonify({
                "error": f"Input should have {input_size} features, but got {input_data.shape[1]}"
            }), 400
        
        # Converting the numpy array to a PyTorch tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_tensor) 
            _, predicted = torch.max(outputs, 1) 

        predicted_class = class_names[int(predicted.item())]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
