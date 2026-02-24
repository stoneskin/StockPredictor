"""
Standalone script to convert trained LightGBM model to ONNX format.
"""

import os
import sys
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CHECKPOINTS_DIR

def convert_to_onnx():
    """Convert trained LightGBM model to ONNX format."""
    try:
        import onnxmltools
        from skl2onnx.common.data_types import FloatTensorType

        print("="*50)
        print("Loading trained model...")
        print("="*50)

        # Load model
        model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        # Load feature names
        feature_path = os.path.join(MODEL_CHECKPOINTS_DIR, "feature_names.txt")
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Feature names loaded: {len(feature_names)} features")

        print("\n" + "="*50)
        print("Converting to ONNX...")
        print("="*50)

        # Convert using onnxmltools with proper data types
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)

        # Save ONNX model
        onnx_path = os.path.join(MODEL_CHECKPOINTS_DIR, "..", "onnx", "model.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        onnxmltools.utils.save_model(onnx_model, onnx_path)

        print(f"✅ ONNX model saved to: {onnx_path}")
        return onnx_path

    except ImportError as e:
        print(f"❌ Import error. Error: {e}")
        print("Install with: pip install onnxmltools skl2onnx")
        return None
    except FileNotFoundError as e:
        print(f"❌ Model file not found. Error: {e}")
        print(f"Make sure model is trained and saved in: {MODEL_CHECKPOINTS_DIR}")
        return None

if __name__ == "__main__":
    onnx_path = convert_to_onnx()
