"""
Random Forest Classifier for Tier 1 (Presence Detection)

Binary classification: Empty room (0) vs Occupied room (1)

Author: Aashik Mathew
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from typing import Tuple, Dict
import yaml


class PresenceDetector:
    """
    Random Forest classifier for presence detection.
    
    Attributes:
        model: Trained Random Forest model
        config: Model configuration
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize presence detector.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config['tier1']['random_forest']
        else:
            self.config = self._default_config()
        
        self.model = RandomForestClassifier(**self.config)
        self.is_trained = False
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Training Random Forest with {X_train.shape[0]} samples...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_acc
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred)
            val_recall = recall_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred)
            
            metrics.update({
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")
            print(f"Validation F1: {val_f1:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict presence for new samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of presence.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Probabilities, shape (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5
    ) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importances_
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test classifier
    print("Testing Presence Detector...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 12
    
    # Empty room: low variance features
    X_empty = np.random.randn(n_samples // 2, n_features) * 0.5
    y_empty = np.zeros(n_samples // 2)
    
    # Occupied room: high variance features
    X_occupied = np.random.randn(n_samples // 2, n_features) * 2.0 + 1.0
    y_occupied = np.ones(n_samples // 2)
    
    # Combine
    X = np.vstack([X_empty, X_occupied])
    y = np.hstack([y_empty, y_occupied])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train model
    detector = PresenceDetector()
    metrics = detector.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_metrics = detector.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {test_metrics['confusion_matrix']}")
    
    # Test prediction
    print(f"\nTesting prediction...")
    test_sample = X_test[:5]
    predictions = detector.predict(test_sample)
    probabilities = detector.predict_proba(test_sample)
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    print("\n✅ Presence detector working correctly!")

