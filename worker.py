"""
CIC-IDS2017 Anomaly Detection Worker
This module loads a pre-trained Isolation Forest model and provides prediction functionality.
"""

import joblib
import numpy as np
import pandas as pd
import sys
import os
from typing import Union, List, Dict, Tuple


class AnomalyDetectionWorker:
    """
    Worker class for anomaly detection predictions using Isolation Forest model.

    Attributes:
        model_path (str): Path to the saved model.pkl file
        model_package (dict): Loaded model package containing model and scaler
        iso_forest: Trained Isolation Forest model
        scaler: StandardScaler for feature normalization
        feature_names (list): Names of features used for training
    """

    def __init__(self, model_path: str = 'model.pkl'):
        """
        Initialize the AnomalyDetectionWorker with a pre-trained model.

        Args:
            model_path (str): Path to the saved model.pkl file

        Raises:
            FileNotFoundError: If model_path does not exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.model_package = joblib.load(model_path)

        self.iso_forest = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.feature_names = self.model_package['feature_names']

        print(f"[+] Model loaded successfully from {model_path}")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Model type: {type(self.iso_forest).__name__}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Union[np.ndarray, List]]:
        """
        Make anomaly detection predictions on input data.

        Args:
            data (pd.DataFrame or np.ndarray): Input data for prediction
                - If DataFrame: must contain all required feature columns
                - If ndarray: must have shape (n_samples, n_features)

        Returns:
            dict: Dictionary containing:
                - 'predictions': Array of predictions (-1 for anomaly, 1 for normal)
                - 'predictions_binary': Array with 1 for anomaly, 0 for normal
                - 'anomaly_scores': Array of anomaly scores
                - 'anomaly_indices': Indices of detected anomalies
                - 'anomaly_count': Number of detected anomalies
                - 'anomaly_percentage': Percentage of anomalies

        Raises:
            ValueError: If input data format is invalid or features don't match
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Input has {data.shape[1]} features but model expects {len(self.feature_names)}"
                )
            data = pd.DataFrame(data, columns=self.feature_names)

        elif isinstance(data, pd.DataFrame):
            # Check if all required features are present
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Select only required features in the correct order
            data = data[self.feature_names]

        else:
            raise ValueError("Input data must be pandas DataFrame or numpy array")

        # Scale the features
        data_scaled = self.scaler.transform(data)

        # Make predictions
        predictions = self.iso_forest.predict(data_scaled)
        anomaly_scores = self.iso_forest.score_samples(data_scaled)

        # Convert to binary format (1 = anomaly, 0 = normal)
        predictions_binary = (predictions == -1).astype(int)

        # Get anomaly indices
        anomaly_indices = np.where(predictions == -1)[0]

        results = {
            'predictions': predictions,
            'predictions_binary': predictions_binary,
            'anomaly_scores': anomaly_scores,
            'anomaly_indices': anomaly_indices,
            'anomaly_count': len(anomaly_indices),
            'anomaly_percentage': (len(anomaly_indices) / len(predictions)) * 100,
            'total_samples': len(predictions)
        }

        return results

    def predict_single(self, data: Union[dict, pd.Series, np.ndarray]) -> Dict[str, Union[int, float]]:
        """
        Make a single prediction for one sample.

        Args:
            data: Single sample data (dict, pandas Series, or numpy array)

        Returns:
            dict: Dictionary containing prediction and anomaly score for single sample
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[0] != 1:
                raise ValueError("Expected single sample (1 row)")

        results = self.predict(data)

        return {
            'prediction': int(results['predictions'][0]),
            'is_anomaly': bool(results['predictions_binary'][0]),
            'anomaly_score': float(results['anomaly_scores'][0])
        }

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            dict: Model information including type, parameters, and feature count
        """
        return {
            'model_type': type(self.iso_forest).__name__,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'contamination': getattr(self.iso_forest, 'contamination', None),
            'n_estimators': getattr(self.iso_forest, 'n_estimators', None),
            'random_state': getattr(self.iso_forest, 'random_state', None)
        }


# Example usage function
def example_usage():
    """Demonstrate how to use the AnomalyDetectionWorker."""

    # Initialize the worker
    worker = AnomalyDetectionWorker('model.pkl')

    # Print model information
    print("\nModel Information:")
    print(worker.get_model_info())

    # Example 1: Predict on a single row
    print("\n" + "="*80)
    print("Example 1: Single Prediction")
    print("="*80)
    sample_data = {col: 0 for col in worker.feature_names}  # Create a sample with zeros
    single_result = worker.predict_single(sample_data)
    print(f"Single sample prediction: {single_result}")

    # Example 2: Predict on multiple rows
    print("\n" + "="*80)
    print("Example 2: Batch Prediction")
    print("="*80)
    batch_data = pd.DataFrame(
        np.random.randn(10, len(worker.feature_names)),
        columns=worker.feature_names
    )
    batch_results = worker.predict(batch_data)
    print(f"Batch predictions summary:")
    print(f"  - Total samples: {batch_results['total_samples']}")
    print(f"  - Anomalies detected: {batch_results['anomaly_count']}")
    print(f"  - Anomaly percentage: {batch_results['anomaly_percentage']:.2f}%")


if __name__ == "__main__":
    example_usage()
