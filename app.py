from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load saved models and scaler
try:
    with open('ridge.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    with open('region_mapping.pkl', 'rb') as f:
        region_mapping = pickle.load(f)
    
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")

# Model metadata
MODEL_METADATA = {
    'algorithm': 'Ridge Regression',
    'optimal_alpha': 0.8302,
    'r2_score': 0.9829,
    'mae': 1.6154,
    'rmse': 2.0807,
    'test_accuracy': '98.29%',
    'samples_trained': 400,
    'features': 8
}


@app.route('/')
def home():
    """Render the home page with the prediction interface."""
    return render_template('index.html', 
                         regions=list(region_mapping.keys()),
                         model_info=MODEL_METADATA)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for FWI prediction.
    
    Expected JSON payload:
    {
        "temperature": float,
        "humidity": float,
        "wind_speed": float,
        "rain": float,
        "ffmc": float,
        "dmc": float,
        "isi": float,
        "region": str
    }
    
    Returns:
    {
        "success": bool,
        "prediction": float,
        "confidence": str,
        "risk_level": str,
        "input_data": dict,
        "timestamp": str
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        required_fields = ['temperature', 'humidity', 'wind_speed', 'rain', 
                          'ffmc', 'dmc', 'isi', 'region']
        
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': f'Missing required fields. Expected: {required_fields}'
            }), 400
        
        # Extract and validate values
        try:
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            wind_speed = float(data['wind_speed'])
            rain = float(data['rain'])
            ffmc = float(data['ffmc'])
            dmc = float(data['dmc'])
            isi = float(data['isi'])
            region = str(data['region']).strip()
            
            # Validate ranges
            if not (0 <= temperature <= 50):
                raise ValueError("Temperature must be between 0 and 50°C")
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity must be between 0 and 100%")
            if not (0 <= wind_speed <= 50):
                raise ValueError("Wind speed must be between 0 and 50 km/h")
            if rain < 0:
                raise ValueError("Rain cannot be negative")
            if not (0 <= ffmc <= 100):
                raise ValueError("FFMC must be between 0 and 100")
            if not (0 <= dmc <= 300):
                raise ValueError("DMC must be between 0 and 300")
            if not (0 <= isi <= 50):
                raise ValueError("ISI must be between 0 and 50")
            if region not in region_mapping:
                raise ValueError(f"Region must be one of: {list(region_mapping.keys())}")
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        
        # Encode region
        region_encoded = region_mapping[region]
        
        # Prepare input array
        input_array = np.array([
            [temperature, humidity, wind_speed, rain, ffmc, dmc, isi, region_encoded]
        ])
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        fwi_prediction = model.predict(input_scaled)[0]
        
        # Determine risk level and confidence
        if fwi_prediction < 10:
            risk_level = "LOW"
            risk_color = "green"
            confidence = "Very High"
        elif fwi_prediction < 25:
            risk_level = "MODERATE"
            risk_color = "yellow"
            confidence = "High"
        elif fwi_prediction < 50:
            risk_level = "HIGH"
            risk_color = "orange"
            confidence = "High"
        else:
            risk_level = "EXTREME"
            risk_color = "red"
            confidence = "Very High"
        
        # Prepare response
        response = {
            'success': True,
            'prediction': round(fwi_prediction, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': confidence,
            'input_data': {
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'rain': rain,
                'ffmc': ffmc,
                'dmc': dmc,
                'isi': isi,
                'region': region
            },
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'algorithm': 'Ridge Regression',
                'test_r2_score': 0.9829,
                'test_mae': 1.6154
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model."""
    return jsonify({
        'success': True,
        'model_info': MODEL_METADATA,
        'features': feature_cols,
        'regions': list(region_mapping.keys()),
        'model_loaded': MODEL_LOADED
    }), 200


@app.route('/api/example-predictions', methods=['GET'])
def example_predictions():
    """Return example predictions for various scenarios."""
    examples = [
        {
            'name': 'Low Fire Risk (Cool, Humid)',
            'data': {
                'temperature': 20, 'humidity': 80, 'wind_speed': 5,
                'rain': 2, 'ffmc': 30, 'dmc': 100, 'isi': 10, 'region': 'Bejaia'
            }
        },
        {
            'name': 'High Fire Risk (Hot, Dry, Windy)',
            'data': {
                'temperature': 35, 'humidity': 30, 'wind_speed': 15,
                'rain': 0, 'ffmc': 90, 'dmc': 200, 'isi': 40, 'region': 'Sidi-Bel-Abbes'
            }
        },
        {
            'name': 'Moderate Fire Risk',
            'data': {
                'temperature': 28, 'humidity': 55, 'wind_speed': 10,
                'rain': 1, 'ffmc': 50, 'dmc': 150, 'isi': 25, 'region': 'Bejaia'
            }
        }
    ]
    
    # Generate predictions for each example
    results = []
    for example in examples:
        try:
            region_encoded = region_mapping[example['data']['region']]
            input_array = np.array([[
                example['data']['temperature'],
                example['data']['humidity'],
                example['data']['wind_speed'],
                example['data']['rain'],
                example['data']['ffmc'],
                example['data']['dmc'],
                example['data']['isi'],
                region_encoded
            ]])
            input_scaled = scaler.transform(input_array)
            fwi = model.predict(input_scaled)[0]
            
            results.append({
                'name': example['name'],
                'data': example['data'],
                'fwi_prediction': round(fwi, 2)
            })
        except Exception as e:
            pass
    
    return jsonify({
        'success': True,
        'examples': results
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    if not MODEL_LOADED:
        print("WARNING: Model files not found. Please ensure ridge.pkl, scaler.pkl, etc. are in the same directory.")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
