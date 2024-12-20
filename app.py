from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load('diabetes_predictor.pkl')  # Ensure the model is trained with the new dataset

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json  # Expected input: {"BMI": 25.0, "Smoker": 1, ..., "Income": 50000}
    
    try:
        # Extract input features from JSON
        features = [
            data['HighBP'],
            data['HighChol'],
            data['CholCheck'],
            data['BMI'],
            data['Smoker'],
            data['Stroke'],
            data['HeartDiseaseorAttack'],
            data['PhysActivity'],
            data['Fruits'],
            data['Veggies'],
            data['HvyAlcoholConsump'],
            data['AnyHealthcare'],
            data['NoDocbcCost'],
            data['GenHlth'],
            data['MentHlth'],
            data['PhysHlth'],
            data['DiffWalk'],
            data['Sex'],
            data['Age'],
            data['Education'],
            data['Income']
        ]

        # Convert features to a numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features_array)
        prediction_label = "Has Diabetes" if prediction[0] == 1 else "No Diabetes"

        # Respond with the prediction
        return jsonify({'Prediction': prediction_label})
    
    except KeyError as e:
        return jsonify({'error': f'Missing key in input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

