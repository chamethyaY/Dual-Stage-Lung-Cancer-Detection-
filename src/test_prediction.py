import os
import numpy as np
import pandas as pd

# Suppress some noisy TensorFlow warnings in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Define our paths
BASE_PATH = r'C:\Users\User\Desktop\Lung Cancer Detection'
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'lung_cancer_3d_cnn.h5')
CSV_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'labels.csv')
PATCH_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'patches')

def test_local_prediction():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find the model file at {MODEL_PATH}")
        return

    print("Loading trained model...")
    # Load the model you downloaded from Colab
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")

    # Load labels to find a real test patch
    df = pd.read_csv(CSV_PATH)
    
    # We will test on the very first patch in our dataset
    sample_row = df.iloc[0]
    patch_filename = sample_row['patch_file']
    actual_label = sample_row['label']
    
    print(f"Testing on patch: {patch_filename}")
    print(f"Actual Ground Truth Label: {'Malignant (1)' if actual_label == 1 else 'Benign (0)'}")

    # Load the actual 3D numpy array
    patch_file_path = os.path.join(PATCH_PATH, patch_filename)
    patch = np.load(patch_file_path)

    # The model expects a shape of (batch_size, 64, 64, 64, channels)
    # Our raw patch is just (64, 64, 64), so we add the batch and channel dimensions
    patch_ready = np.expand_dims(np.expand_dims(patch, axis=0), axis=-1).astype(np.float32)

    print("\nMaking prediction...")
    # Get the raw probability score
    prediction_prob = model.predict(patch_ready, verbose=0)[0][0]

    print(f"Model Prediction Score: {prediction_prob * 100:.2f}% chance of being Malignant")
    
    if prediction_prob > 0.5:
        print("Model Decision: 🚨 It thinks this is Malignant (Cancer)")
    else:
        print("Model Decision: ✅ It thinks this is Benign (Safe)")

if __name__ == "__main__":
    try:
        test_local_prediction()
    except Exception as e:
        print(f"An error occurred: {e}")
