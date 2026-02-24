"""
SageMaker training and deployment script for V2 Classification Models.
Supports multi-horizon prediction with ensemble models.
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
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config_v2 import (
    MODEL_RESULTS_DIR, HORIZONS, DEFAULT_HORIZON,
    MODEL_PARAMS, ENSEMBLE_WEIGHTS
)

# SageMaker configuration (add to config_v2.py if needed)
SAGEMAKER_CONFIG = {
    'role': None,  # Will use execution role if None
    'region': 'us-east-1',
    'framework_version': '1.2-1',
    'py_version': 'py3',
    'instance_count': 1,
    'instance_type': 'ml.m5.large'
}


class CSVSerializer:
    """Serializer for CSV data for inference."""
    content_type = "text/csv"
    def serialize(self, data):
        import numpy as np
        return ",".join(map(str, data.reshape(-1))) + "\n"


class JSONSerializer:
    """Serializer for JSON data for inference."""
    content_type = "application/json"
    def serialize(self, data):
        return json.dumps(data) + "\n"


def upload_data_to_s3(local_path, s3_prefix, sagemaker_session):
    """Upload local dataset to S3."""
    print(f"Uploading {local_path} to S3 bucket {sagemaker_session.default_bucket()}...")
    s3_path = sagemaker_session.upload_data(
        path=local_path,
        key_prefix=s3_prefix,
    )
    print(f"Uploaded to: {s3_path}")
    return s3_path


def prepare_v2_training_data():
    """
    Prepare training data for V2 models.
    Creates CSV files with features and labels for each horizon.
    """
    from data_preparation_v2 import prepare_data
    
    print("\n" + "="*60)
    print("Preparing V2 Training Data for SageMaker")
    print("="*60)
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "data", "sagemaker")
    os.makedirs(output_dir, exist_ok=True)
    
    for horizon in HORIZONS:
        print(f"\nPreparing data for {horizon}-day horizon...")
        
        # Get data
        X, y, feature_names, df = prepare_data(horizon=horizon)
        
        # Create DataFrame
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df['target'] = y
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"train_h{horizon}d.csv")
        feature_df.to_csv(output_path, index=False)
        print(f"  Saved {len(feature_df)} samples to {output_path}")
    
    # Save feature names
    feature_path = os.path.join(output_dir, "feature_names.txt")
    with open(feature_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"\nFeature names saved to {feature_path}")
    
    return output_dir


def train_on_sagemaker_v2():
    """Submit V2 training job to SageMaker."""
    sagemaker_session = sagemaker.Session()
    role = SAGEMAKER_CONFIG['role'] or get_execution_role()
    
    print("="*60)
    print("Starting SageMaker Training - V2 Classification")
    print("="*60)
    
    # Prepare training data
    data_dir = prepare_v2_training_data()
    
    # Upload to S3
    train_s3 = upload_data_to_s3(
        os.path.join(data_dir, f"train_h{DEFAULT_HORIZON}d.csv"),
        "stock-prediction-v2/train",
        sagemaker_session
    )
    
    # Prepare estimator
    sklearn_estimator = SKLearn(
        entry_point="train_v2.py",
        source_dir=os.path.join(PROJECT_ROOT, "src"),
        role=role,
        framework_version=SAGEMAKER_CONFIG['framework_version'],
        py_version=SAGEMAKER_CONFIG['py_version'],
        instance_count=SAGEMAKER_CONFIG['instance_count'],
        instance_type=SAGEMAKER_CONFIG['instance_type'],
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{sagemaker_session.default_bucket()}/stock-prediction-v2/output",
        hyperparameters={
            "horizon": DEFAULT_HORIZON,
            "n_estimators": 100,
            "max_depth": 5
        }
    )
    
    print("\nSubmitting training job...")
    sklearn_estimator.fit({"train": train_s3})
    
    # Deploy
    print("\nDeploying model to endpoint...")
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        serializer=JSONSerializer(),
        deserializer=JSONSerializer()
    )
    
    print(f"\nModel deployed to: {predictor.endpoint_name}")
    print("="*60)
    
    return predictor


def deploy_existing_model_v2(model_path=None, endpoint_name=None):
    """
    Deploy an existing V2 model to SageMaker.
    
    Args:
        model_path: Path to local model file (default: best model in results/v2)
        endpoint_name: Name for the SageMaker endpoint
    """
    sagemaker_session = sagemaker.Session()
    role = SAGEMAKER_CONFIG['role'] or get_execution_role()
    
    if model_path is None:
        # Use best individual model (GradientBoosting)
        model_path = os.path.join(MODEL_RESULTS_DIR, "gradientboosting_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print("="*60)
    print("Deploying V2 Model to SageMaker")
    print("="*60)
    print(f"Model: {model_path}")
    
    # Create model
    model = Model(
        model_data=f"s3://{sagemaker_session.default_bucket()}/models/v2/model.tar.gz",
        role=role,
        framework_model=None,  # Will use sklearn model
        sagemaker_session=sagemaker_session
    )
    
    # Deploy
    if endpoint_name is None:
        endpoint_name = "stock-predictor-v2"
    
    print(f"\nDeploying to endpoint: {endpoint_name}")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONSerializer()
    )
    
    print(f"\nModel deployed successfully!")
    print(f"Endpoint: {predictor.endpoint_name}")
    print("="*60)
    
    return predictor


def predict_v2(predictor, features):
    """
    Make prediction using deployed V2 model.
    
    Args:
        predictor: SageMaker predictor
        features: Feature array or dict
    
    Returns:
        Prediction result
    """
    if isinstance(features, dict):
        payload = json.dumps(features)
    else:
        payload = json.dumps(features.tolist())
    
    result = predictor.predict(payload)
    return result


def cleanup_endpoint(endpoint_name):
    """Delete SageMaker endpoint to avoid charges."""
    import boto3
    sm_client = boto3.client('sagemaker')
    
    print(f"Deleting endpoint: {endpoint_name}")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("Endpoint deleted.")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="SageMaker V2 Training and Deployment")
    parser.add_argument('--mode', choices=['train', 'deploy', 'predict', 'cleanup'], 
                        default='train', help='Operation mode')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model file for deployment')
    parser.add_argument('--endpoint', type=str, default=None,
                        help='Endpoint name for prediction/cleanup')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                        help='Prediction horizon (5, 10, or 20)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        predictor = train_on_sagemaker_v2()
        print(f"\nTraining complete. Endpoint: {predictor.endpoint_name}")
        
    elif args.mode == 'deploy':
        predictor = deploy_existing_model_v2(args.model_path, args.endpoint)
        print(f"\nDeployment complete. Endpoint: {predictor.endpoint_name}")
        
    elif args.mode == 'predict':
        if not args.endpoint:
            print("Error: --endpoint required for predict mode")
            return
        
        # Create predictor
        predictor = Predictor(args.endpoint)
        
        # Example prediction
        # In production, you would compute features from real data
        example_features = [0.0] * 47  # Placeholder - 47 features
        result = predict_v2(predictor, example_features)
        print(f"Prediction result: {result}")
        
    elif args.mode == 'cleanup':
        if args.endpoint:
            cleanup_endpoint(args.endpoint)
        else:
            print("Error: --endpoint required for cleanup mode")


if __name__ == '__main__':
    main()


# Cost estimation (for reference)
"""
SageMaker Costs (us-east-1):
- ml.m5.large: $0.23/hour (training), $0.23/hour (inference)
- Data storage: ~$0.023/GB/month (S3)

Training cost estimate:
- 1 hour training x $0.23 = $0.23
- 1 GB data storage x $0.023 = $0.023/month

Inference cost estimate (24/7):
- 1 instance x $0.23/hour x 24 x 30 = $165.60/month

Recommendation: Use serverless inference or batch transform for cost savings
"""