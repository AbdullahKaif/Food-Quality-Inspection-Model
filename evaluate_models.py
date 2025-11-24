"""
Script to evaluate both models and determine which has the highest validation accuracy.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fix for TensorFlow/Keras compatibility with batch_shape in InputLayer
def patch_input_layer():
    """Patch InputLayer to handle batch_shape in older model configs."""
    try:
        # Try different import paths for different TensorFlow versions
        try:
            from keras.engine.input_layer import InputLayer
        except ImportError:
            try:
                from tensorflow.keras.layers import InputLayer
            except ImportError:
                from keras.layers import InputLayer
        
        original_from_config = InputLayer.from_config
        
        def from_config_patched(cls, config):
            # Create a mutable copy of config
            config = dict(config) if isinstance(config, dict) else config
            # Convert batch_shape to input_shape if present
            if isinstance(config, dict) and 'batch_shape' in config:
                batch_shape = config.get('batch_shape')
                if batch_shape and 'input_shape' not in config:
                    # Convert batch_shape [None, 224, 224, 3] to input_shape [224, 224, 3]
                    config['input_shape'] = batch_shape[1:] if batch_shape[0] is None else batch_shape[1:]
                # Remove batch_shape to avoid errors
                config.pop('batch_shape', None)
            return original_from_config(cls, config)
        
        # Replace the classmethod
        InputLayer.from_config = classmethod(from_config_patched)
        print("InputLayer patch applied successfully")
    except Exception as e:
        print(f"Warning: Could not patch InputLayer: {e}")
        print("Will try alternative method...")

# Apply the patch
patch_input_layer()

# Alternative: Fix model config in H5 file before loading
def fix_model_h5_file(model_path):
    """Fix batch_shape in model H5 file before loading."""
    try:
        import h5py
        import json
        import os
        
        # Make a backup first
        backup_path = model_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                # Read model config
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Recursively fix batch_shape to input_shape
                def fix_config_recursive(obj):
                    if isinstance(obj, dict):
                        # Fix batch_shape in this dict
                        if 'batch_shape' in obj:
                            batch_shape = obj['batch_shape']
                            if batch_shape and 'input_shape' not in obj:
                                # Remove batch dimension (first None)
                                if batch_shape[0] is None:
                                    obj['input_shape'] = batch_shape[1:]
                                else:
                                    obj['input_shape'] = batch_shape[1:]
                            # Remove batch_shape to avoid errors
                            del obj['batch_shape']
                        # Recursively fix nested structures
                        for key, value in obj.items():
                            if isinstance(value, (dict, list)):
                                fix_config_recursive(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                fix_config_recursive(item)
                
                fix_config_recursive(config)
                
                # Write back fixed config
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print(f"Fixed config in {model_path}")
                return True
            else:
                print(f"No model_config found in {model_path}")
                return False
    except Exception as e:
        import traceback
        print(f"Could not fix H5 file {model_path}: {e}")
        print(traceback.format_exc())
        return False

# Configuration (matching the notebook)
DataSet_Directory = r'Food Quality Inspection DataSet\FRUIT-16K'
Image_Size = (224, 224)
Batch_Size = 32
Seed = 4

# Create validation generator
Validation_DataGen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

Validation_Gen = Validation_DataGen.flow_from_directory(
    DataSet_Directory,
    target_size=Image_Size,
    batch_size=Batch_Size,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=Seed
)

print(f"Found {Validation_Gen.samples} validation images")
print(f"Class indices: {Validation_Gen.class_indices}")

# Load both models
print("\nLoading models...")

# Try to fix H5 files first
print("Attempting to fix model config files...")
fix_model_h5_file('Food_Quality_Inspection_Model.h5')
fix_model_h5_file('best_hybrid_hsi_model.h5')

# Try loading with safe_mode=False for older model compatibility
try:
    model1 = tf.keras.models.load_model('Food_Quality_Inspection_Model.h5', compile=False, safe_mode=False)
    print("Loaded Food_Quality_Inspection_Model.h5")
except Exception as e1:
    print(f"First attempt failed: {e1}")
    # Try with safe_mode=True (default in newer versions)
    try:
        model1 = tf.keras.models.load_model('Food_Quality_Inspection_Model.h5', compile=False, safe_mode=True)
        print("Loaded Food_Quality_Inspection_Model.h5 (with safe_mode=True)")
    except Exception as e2:
        print(f"Error loading Food_Quality_Inspection_Model.h5: {e2}")
        raise Exception(f"Failed to load Food_Quality_Inspection_Model.h5: {e1}, {e2}")

try:
    model2 = tf.keras.models.load_model('best_hybrid_hsi_model.h5', compile=False, safe_mode=False)
    print("Loaded best_hybrid_hsi_model.h5")
except Exception as e1:
    print(f"First attempt failed: {e1}")
    # Try with safe_mode=True
    try:
        model2 = tf.keras.models.load_model('best_hybrid_hsi_model.h5', compile=False, safe_mode=True)
        print("Loaded best_hybrid_hsi_model.h5 (with safe_mode=True)")
    except Exception as e2:
        print(f"Error loading best_hybrid_hsi_model.h5: {e2}")
        raise Exception(f"Failed to load best_hybrid_hsi_model.h5: {e1}, {e2}")

# Recompile models for evaluation
model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\nEvaluating Food_Quality_Inspection_Model.h5...")
results1 = model1.evaluate(Validation_Gen, verbose=1)
val_loss1 = results1[0]
val_accuracy1 = results1[1]

# Reset generator for second evaluation
Validation_Gen.reset()

print("\nEvaluating best_hybrid_hsi_model.h5...")
results2 = model2.evaluate(Validation_Gen, verbose=1)
val_loss2 = results2[0]
val_accuracy2 = results2[1]

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"Food_Quality_Inspection_Model.h5:")
print(f"  Validation Accuracy: {val_accuracy1:.4f} ({val_accuracy1*100:.2f}%)")
print(f"  Validation Loss: {val_loss1:.4f}")
print(f"\nbest_hybrid_hsi_model.h5:")
print(f"  Validation Accuracy: {val_accuracy2:.4f} ({val_accuracy2*100:.2f}%)")
print(f"  Validation Loss: {val_loss2:.4f}")

# Determine best model
if val_accuracy1 > val_accuracy2:
    best_model = 'Food_Quality_Inspection_Model.h5'
    best_accuracy = val_accuracy1
    print(f"\n✓ BEST MODEL: {best_model}")
    print(f"  Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
else:
    best_model = 'best_hybrid_hsi_model.h5'
    best_accuracy = val_accuracy2
    print(f"\n✓ BEST MODEL: {best_model}")
    print(f"  Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

print("="*60)

# Save the best model name to a file
with open('best_model.txt', 'w') as f:
    f.write(best_model)

print(f"\nBest model name saved to 'best_model.txt'")
