#!/usr/bin/env python
"""
Model Training Script for WiFi-Sense

Train models for different tiers.

Usage:
    python scripts/train_model.py --tier 1 --data-dir data/raw/presence
    
Author: Aashik Mathew
"""

import argparse
import os
import sys
import numpy as np
import yaml
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processing.preprocessing import CSIPreprocessor
from processing.features import CSIFeatureExtractor
from models.random_forest import PresenceDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Train WiFi-Sense models')
    parser.add_argument('--tier', type=int, default=1, choices=[1, 2, 3],
                       help='Tier level (1, 2, or 3)')
    parser.add_argument('--data-dir', type=str, default='data/raw/presence',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-size', type=float, default=0.15,
                       help='Test set size (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                       help='Validation set size (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def train_tier1(args):
    """Train Tier 1 model (presence detection)."""
    print("=" * 70)
    print("TRAINING TIER 1: PRESENCE DETECTION")
    print("=" * 70)
    
    # Load data
    print(f"\n📂 Loading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    
    csi_data = np.load(data_dir / 'csi_data.npy')
    labels = np.load(data_dir / 'labels.npy')
    
    print(f"✓ CSI data shape: {csi_data.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"  - Class 0 (empty): {np.sum(labels == 0)}")
    print(f"  - Class 1 (occupied): {np.sum(labels == 1)}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Preprocessing
    print(f"\n🔧 Preprocessing CSI data...")
    preprocessor = CSIPreprocessor(
        **config['data']['preprocessing']
    )
    
    csi_processed = preprocessor.process_batch(csi_data)
    print(f"✓ Preprocessed data shape: {csi_processed.shape}")
    
    # Feature extraction
    print(f"\n🔍 Extracting features...")
    extractor = CSIFeatureExtractor(
        feature_names=config['tier1']['features']
    )
    
    features = extractor.extract_batch(csi_processed)
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Number of features: {features.shape[1]}")
    print(f"✓ Feature names: {extractor.get_feature_names()}")
    
    # Split data
    print(f"\n✂️  Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=(args.test_size + args.val_size),
        random_state=args.seed,
        stratify=labels
    )
    
    val_ratio = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=args.seed,
        stratify=y_temp
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Train model
    print(f"\n🤖 Training Random Forest model...")
    detector = PresenceDetector(config_path=args.config)
    train_metrics = detector.train(X_train, y_train, X_val, y_val)
    
    print(f"\n✓ Training complete!")
    print(f"  Train accuracy: {train_metrics['train_accuracy']:.4f}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_metrics = detector.evaluate(X_test, y_test)
    
    print(f"\n{'=' * 70}")
    print(f"TEST SET RESULTS:")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = test_metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"               Empty  Occupied")
    print(f"  Actual Empty    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"      Occupied    {cm[1][0]:4d}    {cm[1][1]:4d}")
    print(f"{'=' * 70}")
    
    # Feature importance
    print(f"\n🎯 Feature Importance:")
    feature_importance = detector.get_feature_importance()
    feature_names = extractor.get_feature_names()
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    
    for i, idx in enumerate(indices[:10]):  # Top 10
        print(f"  {i+1:2d}. {feature_names[idx]:25s}: {feature_importance[idx]:.4f}")
    
    # Save model
    print(f"\n💾 Saving model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'tier1_presence_detector.pkl'
    detector.save(str(model_path))
    
    # Save feature extractor config
    feature_config_path = output_dir / 'tier1_features.yaml'
    with open(feature_config_path, 'w') as f:
        yaml.dump({'features': extractor.get_feature_names()}, f)
    print(f"✓ Feature config saved to {feature_config_path}")
    
    # Save metrics
    all_metrics = {
        **train_metrics,
        **test_metrics,
        'feature_importance': {
            name: float(imp) for name, imp in zip(feature_names, feature_importance)
        }
    }
    
    metrics_path = output_dir / 'tier1_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ TIER 1 TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    
    return test_metrics


def train_tier2(args):
    """Train Tier 2 model (activity recognition)."""
    print("⚠️  Tier 2 training not yet implemented.")
    print("    Complete Tier 1 first, then implement Tier 2.")


def train_tier3(args):
    """Train Tier 3 model (identity recognition)."""
    print("⚠️  Tier 3 training not yet implemented.")
    print("    Complete Tier 2 first, then implement Tier 3.")


def main():
    args = parse_args()
    
    print(f"\n🚀 WiFi-Sense Model Training")
    print(f"{'=' * 70}\n")
    
    if args.tier == 1:
        train_tier1(args)
    elif args.tier == 2:
        train_tier2(args)
    elif args.tier == 3:
        train_tier3(args)


if __name__ == "__main__":
    main()

