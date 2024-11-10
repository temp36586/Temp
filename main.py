from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVC model
model = pickle.load(open('models/svc.pkl', 'rb'))

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission and return results
@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptoms from the form input
    symptoms = request.form.get('symptoms')

    if symptoms:
        # Preprocess the symptoms (this step depends on how your model was trained)
        # You might need to convert this into a feature vector or match it with your dataset
        
        # Assuming symptoms are space-separated and match some columns in the dataset:
        symptom_list = symptoms.split(",")  # Convert input string to list
        
        # Example preprocessing: Convert symptoms into a vector of 0s and 1s based on the presence of symptoms
        # You need to map symptoms to model's expected input here
        # Assuming the model takes a list of binary features representing symptom presence:
        input_data = np.zeros(132)  # Assuming model expects 132 binary inputs
        symptom_indices = []  # List of indices to represent the symptoms

        # Example: Map each symptom to a feature index, e.g., if "fever" is the 10th feature, set input_data[10] = 1
        # For simplicity, skipping this mapping. Replace with your logic

        # Predict using the model
        prediction = model.predict([input_data])[0]

        # Pass the predicted disease to the template
        return render_template('index.html', predicted_disease=prediction)

    else:
        return render_template('index.html', message="No symptoms provided!")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
