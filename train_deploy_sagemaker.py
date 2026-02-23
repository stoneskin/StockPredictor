"""
SageMaker training and deployment script for the stock prediction model.
"""

import os
import sys
import argparse
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAGEMAKER_CONFIG, MODEL_CHECKPOINTS_DIR

class CSVSerializer:
    """Serializer for CSV data for inference."""
    content_type = "text/csv"
    def serialize(self, data):
        import numpy as np
        return ",".join(map(str, data.reshape(-1))) + "\n"

def upload_data_to_s3(local_path, s3_prefix, sagemaker_session):
    """Upload local dataset to S3."""
    print(f"Uploading {local_path} to S3 bucket {sagemaker_session.default_bucket()}...")
    s3_path = sagemaker_session.upload_data(
        path=local_path,
        key_prefix=s3_prefix,
    )
    print(f"Uploaded to: {s3_path}")
    return s3_path

def train_on_sagemaker():
    """Submit training job to SageMaker."""
    sagemaker_session = sagemaker.Session()
    role = SAGEMAKER_CONFIG['role'] or get_execution_role()
    region = SAGEMAKER_CONFIG['region']

    print("="*60)
    print("Starting SageMaker Training")
    print("="*60)

    # Define paths
    train_local = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
    val_local = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "val.csv")
    test_local = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv")

    # Upload to S3
    train_s3 = upload_data_to_s3(train_local, "stock-prediction/train", sagemaker_session)
    val_s3 = upload_data_to_s3(val_local, "stock-prediction/val", sagemaker_session)
    test_s3 = upload_data_to_s3(test_local, "stock-prediction/test", sagemaker_session)

    # Prepare estimator
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir=os.path.join(os.path.dirname(__file__), "src"),
        role=role,
        framework_version=SAGEMAKER_CONFIG['framework_version'],
        py_version=SAGEMAKER_CONFIG['py_version'],
        instance_count=SAGEMAKER_CONFIG['instance_count'],
        instance_type=SAGEMAKER_CONFIG['instance_type'],
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{sagemaker_session.default_bucket()}/stock-prediction/output",
        hyperparameters={
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31
        }
    )

    print("\nSubmitting training job...")
    sklearn_estimator.fit({"train": train_s3, "validation": val_s3})

    # Deploy
    print("\nDeploying model to endpoint...")
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        serializer=CSVSerializer(),
        deserializer=None
    )

    print(f"\nEndpoint name: {predictor.endpoint_name}")
    print("Deployment complete!")

    # Test
    test_df = pd.read_csv(test_local)
    X_test = test_df.drop(columns=[c for c in test_df.columns if c.startswith('target_')]).iloc[0:5]
    csv_input = ",".join(map(str, X_test.values[0])) + "\n"
    result = predictor.predict(csv_input)
    print("Test prediction:", result)

    return predictor.endpoint_name

def deploy_existing_model(model_path):
    """Deploy an existing local model to SageMaker."""
    sagemaker_session = sagemaker.Session()
    role = SAGEMAKER_CONFIG['role'] or get_execution_role()
    region = SAGEMAKER_CONFIG['region']

    print("="*60)
    print("Deploying Existing Model to SageMaker")
    print("="*60)

    # Load model to check
    import joblib
    model = joblib.load(model_path)
    print(f"Loaded model: {type(model).__name__}")

    # Create Model object
    sm_model = Model(
        model_data=None,  # Need to package as tar.gz for BYOM; this is simplified
        role=role,
        framework_version=SAGEMAKER_CONFIG['framework_version'],
        py_version=SAGEMAKER_CONFIG['py_version'],
        sagemaker_session=sagemaker_session,
        entry_point="inference.py",  # Custom inference script
        source_dir=os.path.join(os.path.dirname(__file__), "src")
    )

    # For BYOM (Bring Your Own Model), we need to package model as tar.gz and upload to S3
    # This is a simplified version; in practice use SKLearnModel.from_joblib or package manually
    print("\nNote: Full BYOM deployment requires packaging model.tar.gz with model binary")
    print("Consider using train_on_sagemaker() instead to let SageMaker handle packaging.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Train and Deploy Stock Prediction Model on SageMaker")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "deploy"],
                        help="mode: 'train' to submit training job, 'deploy' to deploy existing model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local model.pkl for deployment (only for deploy mode)")
    args = parser.parse_args()

    if args.mode == "train":
        endpoint_name = train_on_sagemaker()
        print(f"\nEndpoint: {endpoint_name}")
    elif args.mode == "deploy":
        if not args.model_path:
            print("Error: --model-path required for deploy mode")
            return
        deploy_existing_model(args.model_path)

if __name__ == "__main__":
    main()