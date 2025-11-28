"""
Preprocessing Module for CSI Data

This module provides functions for cleaning and normalizing CSI measurements:
- Phase unwrapping (remove 2π discontinuities)
- Outlier detection and removal (Hampel filter)
- Low-pass filtering (Butterworth filter)
- Normalization (z-score, min-max)

Author: Aashik Mathew
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import Tuple, Optional
import warnings


class CSIPreprocessor:
    """
    Preprocess raw CSI data for feature extraction and model training.
    
    The preprocessor handles common CSI measurement issues:
    - Phase wrapping (due to phase being periodic in [-π, π])
    - Outliers (from measurement errors or interference)
    - High-frequency noise (thermal noise, quantization)
    - Scale variations (different environments, distances)
    """
    
    def __init__(
        self,
        enable_phase_unwrap: bool = True,
        enable_outlier_removal: bool = True,
        outlier_threshold: float = 3.0,
        enable_lowpass: bool = True,
        lowpass_cutoff: float = 20.0,
        sample_rate: float = 100.0,
        enable_normalization: bool = True,
        normalization_method: str = "zscore"
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            enable_phase_unwrap: Whether to unwrap phase
            enable_outlier_removal: Whether to remove outliers
            outlier_threshold: Std deviations for outlier detection
            enable_lowpass: Whether to apply low-pass filter
            lowpass_cutoff: Cutoff frequency in Hz
            sample_rate: Sampling rate in Hz
            enable_normalization: Whether to normalize data
            normalization_method: "zscore" or "minmax"
        """
        self.enable_phase_unwrap = enable_phase_unwrap
        self.enable_outlier_removal = enable_outlier_removal
        self.outlier_threshold = outlier_threshold
        self.enable_lowpass = enable_lowpass
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate
        self.enable_normalization = enable_normalization
        self.normalization_method = normalization_method
        
        # Design low-pass filter if needed
        if self.enable_lowpass:
            self.lowpass_filter = self._design_lowpass_filter()
    
    def _design_lowpass_filter(self) -> Tuple:
        """
        Design Butterworth low-pass filter.
        
        Returns:
            Filter coefficients (b, a)
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = self.lowpass_cutoff / nyquist
        
        # Butterworth filter, 4th order
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return b, a
    
    def process(self, csi: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to CSI data.
        
        Args:
            csi: Raw CSI matrix, shape (n_samples, n_subcarriers) - complex
            
        Returns:
            Processed CSI matrix, same shape as input
        """
        csi_processed = csi.copy()
        
        # Extract amplitude and phase
        amplitude = np.abs(csi_processed)
        phase = np.angle(csi_processed)
        
        # 1. Phase unwrapping
        if self.enable_phase_unwrap:
            phase = self._unwrap_phase(phase)
        
        # 2. Outlier removal (on amplitude)
        if self.enable_outlier_removal:
            amplitude = self._remove_outliers(amplitude)
        
        # 3. Low-pass filtering
        if self.enable_lowpass:
            amplitude = self._apply_lowpass(amplitude)
            phase = self._apply_lowpass(phase)
        
        # 4. Normalization
        if self.enable_normalization:
            amplitude = self._normalize(amplitude)
            phase = self._normalize(phase)
        
        # Reconstruct complex CSI
        csi_processed = amplitude * np.exp(1j * phase)
        
        return csi_processed
    
    def _unwrap_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Unwrap phase to remove 2π discontinuities.
        
        Args:
            phase: Phase matrix, shape (n_samples, n_subcarriers)
            
        Returns:
            Unwrapped phase matrix
        """
        # Unwrap along time axis (axis 0) for each subcarrier
        phase_unwrapped = np.unwrap(phase, axis=0)
        return phase_unwrapped
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove outliers using Hampel filter (median-based).
        
        Args:
            data: Data matrix, shape (n_samples, n_subcarriers)
            
        Returns:
            Data with outliers replaced by median values
        """
        data_cleaned = data.copy()
        
        # Process each subcarrier independently
        for i in range(data.shape[1]):
            data_cleaned[:, i] = self._hampel_filter(
                data[:, i], 
                window_size=5, 
                n_sigmas=self.outlier_threshold
            )
        
        return data_cleaned
    
    def _hampel_filter(
        self, 
        x: np.ndarray, 
        window_size: int = 5, 
        n_sigmas: float = 3.0
    ) -> np.ndarray:
        """
        Apply Hampel filter to 1D signal.
        
        The Hampel filter is a robust outlier detector that uses median
        and median absolute deviation (MAD) instead of mean and std.
        
        Args:
            x: 1D signal
            window_size: Window size for median calculation
            n_sigmas: Number of MAD for outlier threshold
            
        Returns:
            Filtered signal
        """
        x_filtered = x.copy()
        k = 1.4826  # Scale factor for MAD to approximate std
        
        half_window = window_size // 2
        n = len(x)
        
        for i in range(half_window, n - half_window):
            # Get window
            window = x[i - half_window:i + half_window + 1]
            
            # Compute median and MAD
            median = np.median(window)
            mad = k * np.median(np.abs(window - median))
            
            # Check if outlier
            if np.abs(x[i] - median) > n_sigmas * mad:
                x_filtered[i] = median
        
        return x_filtered
    
    def _apply_lowpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Args:
            data: Data matrix, shape (n_samples, n_subcarriers)
            
        Returns:
            Filtered data
        """
        data_filtered = data.copy()
        b, a = self.lowpass_filter
        
        # Filter each subcarrier independently
        for i in range(data.shape[1]):
            # Use filtfilt for zero-phase filtering
            data_filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return data_filtered
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data: Data matrix, shape (n_samples, n_subcarriers)
            
        Returns:
            Normalized data
        """
        if self.normalization_method == "zscore":
            # Z-score normalization per subcarrier
            data_normalized = np.zeros_like(data)
            for i in range(data.shape[1]):
                mean = np.mean(data[:, i])
                std = np.std(data[:, i])
                if std > 1e-8:  # Avoid division by zero
                    data_normalized[:, i] = (data[:, i] - mean) / std
                else:
                    data_normalized[:, i] = data[:, i] - mean
        
        elif self.normalization_method == "minmax":
            # Min-max normalization to [0, 1] per subcarrier
            data_normalized = np.zeros_like(data)
            for i in range(data.shape[1]):
                min_val = np.min(data[:, i])
                max_val = np.max(data[:, i])
                if max_val - min_val > 1e-8:
                    data_normalized[:, i] = (data[:, i] - min_val) / (max_val - min_val)
                else:
                    data_normalized[:, i] = 0.5
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return data_normalized
    
    def process_batch(self, csi_batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of CSI samples.
        
        Args:
            csi_batch: Batch of CSI data, shape (n_samples, n_timesteps, n_subcarriers)
            
        Returns:
            Processed batch
        """
        n_samples = csi_batch.shape[0]
        processed_batch = np.zeros_like(csi_batch)
        
        for i in range(n_samples):
            processed_batch[i] = self.process(csi_batch[i])
        
        return processed_batch


def quick_preprocess(csi: np.ndarray, sample_rate: float = 100.0) -> np.ndarray:
    """
    Quick preprocessing with default settings.
    
    Args:
        csi: Raw CSI data
        sample_rate: Sampling rate in Hz
        
    Returns:
        Preprocessed CSI
    """
    preprocessor = CSIPreprocessor(sample_rate=sample_rate)
    return preprocessor.process(csi)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing CSI Preprocessor...")
    
    # Generate synthetic test data
    n_samples = 500
    n_subcarriers = 30
    
    # Create synthetic CSI with noise and outliers
    t = np.linspace(0, 5, n_samples)
    csi_clean = np.zeros((n_samples, n_subcarriers), dtype=complex)
    
    for i in range(n_subcarriers):
        # Sinusoidal signal
        amplitude = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * t)
        phase = 2 * np.pi * 5 * t + i * 0.1
        csi_clean[:, i] = amplitude * np.exp(1j * phase)
    
    # Add noise
    noise = 0.1 * (np.random.randn(n_samples, n_subcarriers) + 
                   1j * np.random.randn(n_samples, n_subcarriers))
    csi_noisy = csi_clean + noise
    
    # Add outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    csi_noisy[outlier_indices, :] *= 5.0
    
    print(f"Raw CSI shape: {csi_noisy.shape}")
    print(f"Raw CSI amplitude - mean: {np.mean(np.abs(csi_noisy)):.4f}, std: {np.std(np.abs(csi_noisy)):.4f}")
    
    # Preprocess
    preprocessor = CSIPreprocessor(
        enable_phase_unwrap=True,
        enable_outlier_removal=True,
        enable_lowpass=True,
        lowpass_cutoff=20.0,
        sample_rate=100.0,
        enable_normalization=True,
        normalization_method="zscore"
    )
    
    csi_processed = preprocessor.process(csi_noisy)
    
    print(f"\nProcessed CSI shape: {csi_processed.shape}")
    print(f"Processed CSI amplitude - mean: {np.mean(np.abs(csi_processed)):.4f}, std: {np.std(np.abs(csi_processed)):.4f}")
    
    # Test batch processing
    csi_batch = np.random.randn(10, 500, 30) + 1j * np.random.randn(10, 500, 30)
    processed_batch = preprocessor.process_batch(csi_batch)
    print(f"\nBatch processing: {csi_batch.shape} -> {processed_batch.shape}")
    
    print("\n✅ Preprocessor working correctly!")

