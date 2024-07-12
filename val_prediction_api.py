from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
# Note: You need to train and save this model beforehand
model = XGBClassifier()
model.load_model('bayesopt_xgb_model.xgb')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json['data']
        
        # Convert the input data to a numpy array
        input_data = np.array(data).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)