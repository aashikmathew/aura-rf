"""
Feature Extraction Module for CSI Data

This module extracts statistical features from preprocessed CSI measurements
for Tier 1 (presence detection). Features capture amplitude, phase, and temporal
characteristics of the CSI signal.

Author: Aashik Mathew
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from scipy.fft import fft


class CSIFeatureExtractor:
    """
    Extract features from CSI data for machine learning models.
    
    For Tier 1 (presence detection), we extract 12 statistical features:
    
    Amplitude Features (5):
    - amplitude_mean: Average signal strength
    - amplitude_std: Variance in signal strength
    - amplitude_max: Peak signal strength
    - amplitude_min: Minimum signal strength
    - amplitude_range: Difference between max and min
    
    Phase Features (3):
    - phase_circular_mean: Average phase (circular statistics)
    - phase_circular_std: Phase variance
    - phase_stability: How stable the phase is over time
    
    Temporal Features (4):
    - autocorr_lag1: First-order autocorrelation (temporal correlation)
    - zero_crossing_rate: Rate of sign changes
    - spectral_entropy: Entropy of frequency spectrum
    - temporal_variance: Variance of amplitude over time
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            feature_names: List of features to extract. If None, extracts all.
        """
        self.all_features = [
            'amplitude_mean',
            'amplitude_std',
            'amplitude_max',
            'amplitude_min',
            'amplitude_range',
            'phase_circular_mean',
            'phase_circular_std',
            'phase_stability',
            'autocorr_lag1',
            'zero_crossing_rate',
            'spectral_entropy',
            'temporal_variance'
        ]
        
        self.feature_names = feature_names if feature_names else self.all_features
    
    def extract(self, csi: np.ndarray) -> np.ndarray:
        """
        Extract features from a single CSI sample.
        
        Args:
            csi: CSI matrix, shape (n_timesteps, n_subcarriers) - complex
            
        Returns:
            Feature vector, shape (n_features,)
        """
        features = {}
        
        # Extract amplitude and phase
        amplitude = np.abs(csi)
        phase = np.angle(csi)
        
        # Amplitude features
        if 'amplitude_mean' in self.feature_names:
            features['amplitude_mean'] = self._amplitude_mean(amplitude)
        
        if 'amplitude_std' in self.feature_names:
            features['amplitude_std'] = self._amplitude_std(amplitude)
        
        if 'amplitude_max' in self.feature_names:
            features['amplitude_max'] = self._amplitude_max(amplitude)
        
        if 'amplitude_min' in self.feature_names:
            features['amplitude_min'] = self._amplitude_min(amplitude)
        
        if 'amplitude_range' in self.feature_names:
            features['amplitude_range'] = self._amplitude_range(amplitude)
        
        # Phase features
        if 'phase_circular_mean' in self.feature_names:
            features['phase_circular_mean'] = self._phase_circular_mean(phase)
        
        if 'phase_circular_std' in self.feature_names:
            features['phase_circular_std'] = self._phase_circular_std(phase)
        
        if 'phase_stability' in self.feature_names:
            features['phase_stability'] = self._phase_stability(phase)
        
        # Temporal features
        if 'autocorr_lag1' in self.feature_names:
            features['autocorr_lag1'] = self._autocorr_lag1(amplitude)
        
        if 'zero_crossing_rate' in self.feature_names:
            features['zero_crossing_rate'] = self._zero_crossing_rate(amplitude)
        
        if 'spectral_entropy' in self.feature_names:
            features['spectral_entropy'] = self._spectral_entropy(amplitude)
        
        if 'temporal_variance' in self.feature_names:
            features['temporal_variance'] = self._temporal_variance(amplitude)
        
        # Convert to array in consistent order
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        return feature_vector
    
    def extract_batch(self, csi_batch: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of CSI samples.
        
        Args:
            csi_batch: Batch of CSI data, shape (n_samples, n_timesteps, n_subcarriers)
            
        Returns:
            Feature matrix, shape (n_samples, n_features)
        """
        n_samples = csi_batch.shape[0]
        n_features = len(self.feature_names)
        
        features = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            features[i] = self.extract(csi_batch[i])
        
        return features
    
    # ========== Amplitude Features ==========
    
    def _amplitude_mean(self, amplitude: np.ndarray) -> float:
        """Average amplitude across all subcarriers and time."""
        return float(np.mean(amplitude))
    
    def _amplitude_std(self, amplitude: np.ndarray) -> float:
        """Standard deviation of amplitude."""
        return float(np.std(amplitude))
    
    def _amplitude_max(self, amplitude: np.ndarray) -> float:
        """Maximum amplitude."""
        return float(np.max(amplitude))
    
    def _amplitude_min(self, amplitude: np.ndarray) -> float:
        """Minimum amplitude."""
        return float(np.min(amplitude))
    
    def _amplitude_range(self, amplitude: np.ndarray) -> float:
        """Range of amplitude (max - min)."""
        return float(np.max(amplitude) - np.min(amplitude))
    
    # ========== Phase Features ==========
    
    def _phase_circular_mean(self, phase: np.ndarray) -> float:
        """
        Circular mean of phase (since phase is periodic).
        
        Uses the formula: arctan2(mean(sin(θ)), mean(cos(θ)))
        """
        mean_sin = np.mean(np.sin(phase))
        mean_cos = np.mean(np.cos(phase))
        circular_mean = np.arctan2(mean_sin, mean_cos)
        return float(circular_mean)
    
    def _phase_circular_std(self, phase: np.ndarray) -> float:
        """
        Circular standard deviation of phase.
        
        Computed from the resultant vector length R:
        circular_std = sqrt(-2 * log(R))
        """
        mean_sin = np.mean(np.sin(phase))
        mean_cos = np.mean(np.cos(phase))
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Avoid log(0)
        R = np.clip(R, 1e-8, 1.0)
        
        circular_std = np.sqrt(-2 * np.log(R))
        return float(circular_std)
    
    def _phase_stability(self, phase: np.ndarray) -> float:
        """
        Phase stability: inverse of phase variance over time.
        
        Higher values indicate more stable phase.
        """
        # Compute phase differences over time for each subcarrier
        phase_diff = np.diff(phase, axis=0)
        
        # Wrap to [-π, π]
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        
        # Stability is inverse of variance (add small constant to avoid division by zero)
        phase_var = np.var(phase_diff)
        stability = 1.0 / (phase_var + 1e-6)
        
        return float(stability)
    
    # ========== Temporal Features ==========
    
    def _autocorr_lag1(self, amplitude: np.ndarray) -> float:
        """
        First-order autocorrelation (correlation with lag-1).
        
        Measures temporal correlation: high for smooth signals, low for noisy.
        """
        # Average across subcarriers
        amp_avg = np.mean(amplitude, axis=1)
        
        # Compute autocorrelation at lag 1
        if len(amp_avg) < 2:
            return 0.0
        
        # Normalize
        amp_centered = amp_avg - np.mean(amp_avg)
        
        autocorr = np.correlate(amp_centered[:-1], amp_centered[1:], mode='valid')[0]
        norm_factor = np.sum(amp_centered[:-1]**2)
        
        if norm_factor > 1e-8:
            autocorr = autocorr / norm_factor
        else:
            autocorr = 0.0
        
        return float(autocorr)
    
    def _zero_crossing_rate(self, amplitude: np.ndarray) -> float:
        """
        Zero crossing rate of amplitude signal.
        
        Computed as the rate at which the signal crosses its mean.
        High values indicate rapid fluctuations.
        """
        # Average across subcarriers
        amp_avg = np.mean(amplitude, axis=1)
        
        # Center around mean
        amp_centered = amp_avg - np.mean(amp_avg)
        
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.sign(amp_centered)))) / 2
        
        # Normalize by length
        zcr = zero_crossings / len(amp_avg)
        
        return float(zcr)
    
    def _spectral_entropy(self, amplitude: np.ndarray) -> float:
        """
        Spectral entropy: entropy of the frequency spectrum.
        
        Low entropy = pure tones (periodic motion like walking)
        High entropy = noise (random or complex motion)
        """
        # Average across subcarriers
        amp_avg = np.mean(amplitude, axis=1)
        
        # Compute power spectrum
        spectrum = np.abs(fft(amp_avg))
        
        # Use only positive frequencies
        spectrum = spectrum[:len(spectrum)//2]
        
        # Normalize to get probability distribution
        power = spectrum ** 2
        power_sum = np.sum(power)
        
        if power_sum > 1e-8:
            power_norm = power / power_sum
            
            # Compute entropy
            # Add small constant to avoid log(0)
            power_norm = power_norm + 1e-10
            entropy = -np.sum(power_norm * np.log2(power_norm))
        else:
            entropy = 0.0
        
        return float(entropy)
    
    def _temporal_variance(self, amplitude: np.ndarray) -> float:
        """
        Temporal variance: variance of amplitude over time.
        
        High values indicate dynamic environment (motion).
        Low values indicate static environment.
        """
        # Compute variance over time for each subcarrier, then average
        temporal_var = np.mean(np.var(amplitude, axis=0))
        
        return float(temporal_var)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names
    
    def get_feature_dict(self, csi: np.ndarray) -> Dict[str, float]:
        """
        Extract features and return as dictionary.
        
        Args:
            csi: CSI matrix
            
        Returns:
            Dictionary mapping feature names to values
        """
        feature_vector = self.extract(csi)
        return dict(zip(self.feature_names, feature_vector))


if __name__ == "__main__":
    # Test feature extraction
    print("Testing CSI Feature Extractor...")
    
    # Generate synthetic test data
    n_samples = 3
    n_timesteps = 500
    n_subcarriers = 30
    
    # Create synthetic CSI
    csi_batch = np.random.randn(n_samples, n_timesteps, n_subcarriers) + \
                1j * np.random.randn(n_samples, n_timesteps, n_subcarriers)
    
    # Extract features
    extractor = CSIFeatureExtractor()
    
    print(f"\nExtracting features from single sample...")
    features_single = extractor.extract(csi_batch[0])
    print(f"Feature vector shape: {features_single.shape}")
    print(f"Number of features: {len(features_single)}")
    
    print(f"\nFeature names and values:")
    feature_dict = extractor.get_feature_dict(csi_batch[0])
    for name, value in feature_dict.items():
        print(f"  {name:25s}: {value:10.6f}")
    
    print(f"\nExtracting features from batch...")
    features_batch = extractor.extract_batch(csi_batch)
    print(f"Feature matrix shape: {features_batch.shape}")
    print(f"Expected: ({n_samples}, {len(extractor.get_feature_names())})")
    
    # Test with subset of features
    print(f"\nTesting with subset of features...")
    subset_features = ['amplitude_mean', 'amplitude_std', 'phase_stability', 'autocorr_lag1']
    extractor_subset = CSIFeatureExtractor(feature_names=subset_features)
    features_subset = extractor_subset.extract(csi_batch[0])
    print(f"Subset feature vector shape: {features_subset.shape}")
    print(f"Features: {extractor_subset.get_feature_names()}")
    
    print("\n✅ Feature extractor working correctly!")

