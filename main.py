from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (assuming model is in models directory)
model = pickle.load(open('models/svc.pkl', 'rb'))

# Home route to render the index page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the symptoms from the form
        symptoms = request.form.get('symptoms')
        
        if symptoms:
            # Placeholder: You will need to preprocess the symptoms based on your model requirements
            # Here we assume some preprocessing, adjust as per your model
            input_data = np.zeros(132)  # Adjust the size according to your model input
            # Convert symptoms to the required format for your model
            
            prediction = model.predict([input_data])[0]

            # Render the result in the template
            return render_template('index.html', predicted_disease=prediction)

        else:
            # If no symptoms were provided
            return render_template('index.html', message="Please enter symptoms.")
    else:
        return render_template('index.html', message="Invalid request.")

if __name__ == "__main__":
    app.run(debug=True)
