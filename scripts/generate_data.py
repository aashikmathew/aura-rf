#!/usr/bin/env python
"""
Data Generation Script for WiFi-Sense

Generate CSI data for training and evaluation.

Usage:
    python scripts/generate_data.py --tier 1 --samples 1000 --seed 42
    
Author: Aashik Mathew
"""

import argparse
import os
import sys
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation.csi_generator import CSIGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Generate WiFi CSI data')
    parser.add_argument('--tier', type=int, default=1, choices=[1, 2, 3],
                       help='Tier level (1, 2, or 3)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of each sample in seconds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for generated data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    return parser.parse_args()


def generate_tier1_data(args):
    """Generate data for Tier 1 (presence detection)."""
    print("=" * 70)
    print("GENERATING TIER 1 DATA: PRESENCE DETECTION")
    print("=" * 70)
    
    # Load configuration
    generator = CSIGenerator(config_path=args.config)
    
    # Calculate number of empty and occupied samples
    n_empty = args.samples // 2
    n_occupied = args.samples - n_empty
    
    print(f"\nConfiguration:")
    print(f"  Total samples: {args.samples}")
    print(f"  Empty room: {n_empty}")
    print(f"  Occupied room: {n_occupied}")
    print(f"  Duration per sample: {args.duration}s")
    print(f"  Seed: {args.seed}")
    
    # Generate data
    print(f"\nGenerating CSI data...")
    csi_data, labels = generator.generate_batch(
        n_empty=n_empty,
        n_occupied=n_occupied,
        duration=args.duration,
        seed=args.seed
    )
    
    print(f"✓ Generated CSI data: {csi_data.shape}")
    print(f"✓ Labels: {labels.shape}")
    print(f"  - Empty samples: {np.sum(labels == 0)}")
    print(f"  - Occupied samples: {np.sum(labels == 1)}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / 'presence'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    csi_path = output_dir / 'csi_data.npy'
    labels_path = output_dir / 'labels.npy'
    
    np.save(csi_path, csi_data)
    np.save(labels_path, labels)
    
    print(f"\n✓ Data saved:")
    print(f"  CSI: {csi_path}")
    print(f"  Labels: {labels_path}")
    
    # Save metadata
    metadata = {
        'tier': 1,
        'n_samples': args.samples,
        'n_empty': n_empty,
        'n_occupied': n_occupied,
        'duration': args.duration,
        'sample_rate': generator.sample_rate,
        'n_subcarriers': generator.n_subcarriers,
        'seed': args.seed,
        'shape': list(csi_data.shape)
    }
    
    metadata_path = output_dir / 'metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"  Metadata: {metadata_path}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ TIER 1 DATA GENERATION COMPLETE!")
    print(f"{'=' * 70}")


def generate_tier2_data(args):
    """Generate data for Tier 2 (activity recognition)."""
    print("⚠️  Tier 2 data generation not yet implemented.")
    print("    Complete Tier 1 first, then implement Tier 2.")


def generate_tier3_data(args):
    """Generate data for Tier 3 (identity recognition)."""
    print("⚠️  Tier 3 data generation not yet implemented.")
    print("    Complete Tier 2 first, then implement Tier 3.")


def main():
    args = parse_args()
    
    print(f"\n🚀 WiFi-Sense Data Generation")
    print(f"{'=' * 70}\n")
    
    if args.tier == 1:
        generate_tier1_data(args)
    elif args.tier == 2:
        generate_tier2_data(args)
    elif args.tier == 3:
        generate_tier3_data(args)


if __name__ == "__main__":
    main()

