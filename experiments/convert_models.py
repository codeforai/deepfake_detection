"""
Convert Keras model to optimized formats:
- TFLite FP32 (full precision)
- TFLite INT8 (quantized - smaller, faster)

Usage:
    python convert_models.py
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000 
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0 
TARGET_TIME_STEPS = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH)  # = 313
def build_cnn_model(input_shape=(40, 312, 1)):
    """Rebuild CNN architecture"""
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout4')(x)
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.5, name='dropout5')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='VoiceClassifierCNN')

def load_original_model():
    """Load the original Keras model"""
    print("\n[1/4] Loading original Keras model...")
    try:
        model = tf.keras.models.load_model("model/best_model_elevenlabs.h5")
        print("Loaded directly")
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Rebuilding from architecture...")
        model = build_cnn_model()
        model.load_weights("model/best_model_elevenlabs.h5")
        print("Loaded via weights")
    return model

def convert_to_tflite_fp32(model):
    """Convert to TFLite FP32 (full precision)"""
    print("\n[2/4] Converting to TFLite FP32...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # No quantization, keep FP32
    tflite_model = converter.convert()
    
    output_path = "model/model_fp32.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"Saved to {output_path} ({size_mb:.2f} MB)")
    return output_path

def representative_dataset_gen():
    """Generate representative dataset for INT8 quantization"""
    # Generate random samples matching input shape
    for _ in range(100):
        data = np.random.randn(1, N_LFCC, TARGET_TIME_STEPS, 1).astype(np.float32)
        yield [data]

def convert_to_tflite_int8(model):
    """Convert to TFLite INT8 (quantized)"""
    print("\n[3/4] Converting to TFLite INT8 (Post-Training Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce INT8 for all ops (if possible)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_model = converter.convert()
        output_path = "model/model_int8.tflite"
    except Exception as e:
        print(f"WARNING: Full INT8 failed: {e}")
        print("Falling back to hybrid quantization...")
        # Fallback to hybrid quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        output_path = "model/model_int8_hybrid.tflite"
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"Saved to {output_path} ({size_mb:.2f} MB)")
    return output_path

def test_models():
    """Test all models work"""
    print("\n[4/4] Testing converted models...")
    
    # Test input
    test_input = np.random.randn(1, N_LFCC, TARGET_TIME_STEPS, 1).astype(np.float32)
    
    models_to_test = [
        ("model/model_fp32.tflite", "TFLite FP32"),
        ("model/model_int8.tflite", "TFLite INT8"),
        ("model/model_int8_hybrid.tflite", "TFLite INT8 Hybrid")
    ]
    
    for model_path, name in models_to_test:
        if not os.path.exists(model_path):
            continue
            
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Handle INT8 input type
            if input_details[0]['dtype'] == np.uint8:
                # Quantize input
                scale, zero_point = input_details[0]['quantization']
                test_input_quant = (test_input / scale + zero_point).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], test_input_quant)
            else:
                interpreter.set_tensor(input_details[0]['index'], test_input)
            
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Handle INT8 output type
            if output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale
            
            print(f"{name}: {output[0][0]:.4f}")
        except Exception as e:
            print(f"FAILED {name}: {e}")

def main():
    print("="*60)
    print("Model Conversion Pipeline")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists("model/best_model_elevenlabs.h5"):
        print("ERROR: model/best_model_elevenlabs.h5 not found!")
        return
    
    # Load original model
    model = load_original_model()
    
    h5_size = os.path.getsize("model/best_model_elevenlabs.h5") / 1024 / 1024
    print(f"Original H5 size: {h5_size:.2f} MB")
    
    # Convert to different formats
    fp32_path = convert_to_tflite_fp32(model)
    int8_path = convert_to_tflite_int8(model)
    
    # Test all models
    test_models()
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print("\nAvailable models:")
    print("  - model/best_model_elevenlabs.h5              (Original Keras)")
    print("  - model/model_fp32.tflite     (TFLite FP32)")
    print("  - model/model_int8*.tflite    (TFLite INT8)")
    print("\nNext: Run benchmark.py to compare performance")

if __name__ == "__main__":
    main()
