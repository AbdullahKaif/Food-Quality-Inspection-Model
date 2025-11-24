"""
Flask backend API for Food Quality Inspection Model.
Serves the model with the highest validation accuracy.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io

# Fix for TensorFlow/Keras compatibility with batch_shape in InputLayer
def patch_input_layer():
    """Patch InputLayer to handle batch_shape in older model configs."""
    try:
        from tensorflow.keras.engine.input_layer import InputLayer
        
        original_from_config = InputLayer.from_config
        
        def from_config_patched(cls, config):
            # Create a copy to avoid modifying the original
            config = dict(config) if isinstance(config, dict) else config
            # Convert batch_shape to input_shape if present
            if isinstance(config, dict) and 'batch_shape' in config:
                if config['batch_shape'] and 'input_shape' not in config:
                    config['input_shape'] = config['batch_shape'][1:]
                # Remove batch_shape to avoid errors
                config.pop('batch_shape', None)
            return original_from_config(cls, config)
        
        # Bind as classmethod
        InputLayer.from_config = classmethod(from_config_patched)
        print("InputLayer patch applied successfully")
    except Exception as e:
        print(f"Warning: Could not patch InputLayer: {e}")

# Apply the patch
patch_input_layer()

def simulate_hsi_from_rgb(rgb_image, num_bands=31):
    """Simulate HSI cube from RGB image (matching notebook function)."""
    height, width = rgb_image.shape[:2]
    hsi_cube = np.zeros((height, width, num_bands), dtype=np.float32)
    
    rgb_wavelengths = np.array([630, 530, 450])
    target_wavelengths = np.linspace(400, 700, num_bands)
    
    for i, target_wl in enumerate(target_wavelengths):
        weights = np.exp(-((rgb_wavelengths - target_wl) ** 2) / (2 * 50 ** 2))
        weights = weights / weights.sum()
        hsi_cube[:, :, i] = np.sum(rgb_image * weights, axis=2)
    
    return hsi_cube

app = Flask(__name__)
CORS(app)

# Load the best model
try:
    with open('best_model.txt', 'r') as f:
        best_model_name = f.read().strip()
except FileNotFoundError:
    # If best_model.txt doesn't exist, evaluate both models
    print("Evaluating models to determine the best one...")
    import subprocess
    subprocess.run(['python', 'evaluate_models.py'], check=True)
    with open('best_model.txt', 'r') as f:
        best_model_name = f.read().strip()

print(f"Loading model: {best_model_name}")
try:
    # Try loading with safe_mode=False first (for older models)
    model = tf.keras.models.load_model(best_model_name, compile=False, safe_mode=False)
    print("Model loaded successfully!")
except Exception as e1:
    print(f"First attempt failed: {e1}")
    print("Attempting alternative loading method...")
    # Try with safe_mode=True (default in newer versions)
    try:
        model = tf.keras.models.load_model(best_model_name, compile=False, safe_mode=True)
        print("Model loaded with safe_mode=True")
    except Exception as e2:
        raise Exception(f"Failed to load model with both methods. First error: {e1}, Second error: {e2}")

# Recompile the model for inference (optional, but helps with compatibility)
try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
except:
    # If compilation fails, that's okay - we can still use for inference
    print("Warning: Could not recompile model, but can still use for inference")

# Check model input structure
print(f"Model inputs: {model.inputs}")
print(f"Number of inputs: {len(model.inputs)}")
for i, inp in enumerate(model.inputs):
    print(f"Input {i}: {inp.name}, shape: {inp.shape}")

# Model configuration
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Fresh', 'Spoiled']

def preprocess_image(image):
    """Preprocess image for model prediction."""
    # Resize image
    image = image.resize(IMAGE_SIZE)
    # Convert to array
    img_array = img_to_array(image)
    # Normalize
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': best_model_name,
        'message': 'Food Quality Inspection API is running'
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    return jsonify({
        'model_name': best_model_name,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'classes': CLASS_NAMES
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict food quality from uploaded image."""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Check model input structure
        num_inputs = len(model.inputs) if isinstance(model.inputs, list) else 1
        input_names = [inp.name for inp in model.inputs] if isinstance(model.inputs, list) else []
        
        print(f"Model has {num_inputs} input(s): {input_names}")
        
        # Make prediction - try different input formats
        predictions = None
        error_messages = []
        
        # Try 1: Dictionary with named inputs (most common for custom models)
        if input_names:
            try:
                if num_inputs > 1 and 'HSI_Input' in input_names:
                    # Model expects both RGB and HSI - simulate HSI
                    hsi_array = simulate_hsi_from_rgb(img_array[0], num_bands=31)
                    hsi_array = np.expand_dims(hsi_array, axis=0)
                    predictions = model.predict({'RGB_Input': img_array, 'HSI_Input': hsi_array}, verbose=0)
                elif 'RGB_Input' in input_names:
                    predictions = model.predict({'RGB_Input': img_array}, verbose=0)
                else:
                    # Try with first input name
                    predictions = model.predict({input_names[0]: img_array}, verbose=0)
            except Exception as e:
                error_messages.append(f"Dictionary input failed: {str(e)}")
        
        # Try 2: List input
        if predictions is None:
            try:
                if num_inputs > 1:
                    if 'HSI_Input' in input_names:
                        # Generate HSI
                        hsi_array = simulate_hsi_from_rgb(img_array[0], num_bands=31)
                        hsi_array = np.expand_dims(hsi_array, axis=0)
                        predictions = model.predict([img_array, hsi_array], verbose=0)
                    else:
                        predictions = model.predict([img_array] * num_inputs, verbose=0)
                else:
                    predictions = model.predict([img_array], verbose=0)
            except Exception as e:
                error_messages.append(f"List input failed: {str(e)}")
        
        # Try 3: Direct array input
        if predictions is None:
            try:
                predictions = model.predict(img_array, verbose=0)
            except Exception as e:
                error_messages.append(f"Direct array input failed: {str(e)}")
        
        if predictions is None:
            raise Exception(f"All prediction methods failed. Errors: {'; '.join(error_messages)}")
        
        # Extract prediction value
        if isinstance(predictions, (list, tuple, np.ndarray)):
            if isinstance(predictions, (list, tuple)):
                pred_value = predictions[0]
            else:
                pred_value = predictions
            
            # Handle numpy array
            if isinstance(pred_value, np.ndarray):
                if pred_value.ndim > 0:
                    prediction = float(pred_value.flatten()[0])
                else:
                    prediction = float(pred_value.item())
            else:
                prediction = float(pred_value)
        else:
            prediction = float(predictions)
        
        # Determine class
        predicted_class = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'probability': {
                'Fresh': round((1 - prediction) * 100, 2),
                'Spoiled': round(prediction * 100, 2)
            }
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details if app.debug else None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
