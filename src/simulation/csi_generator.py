"""
CSI (Channel State Information) Generator Module

This module simulates realistic WiFi CSI data based on physical channel models.
It implements multipath propagation, Doppler effects, and realistic noise models.

Author: Aashik Mathew
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import yaml


class CSIGenerator:
    """
    Generate realistic Channel State Information (CSI) data for WiFi sensing.
    
    The CSI represents the channel frequency response H(f,t) at different subcarriers
    and time instances. It captures amplitude and phase changes caused by:
    - Multipath propagation (reflections, diffraction)
    - Human motion (Doppler shifts, shadowing)
    - Environmental noise (thermal, interference)
    
    Attributes:
        n_subcarriers (int): Number of OFDM subcarriers (typically 30)
        sample_rate (float): Sampling rate in Hz (typically 100 Hz)
        carrier_freq (float): WiFi carrier frequency in Hz (e.g., 5.2 GHz)
        bandwidth (float): Channel bandwidth in Hz (e.g., 20 MHz)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CSI generator with configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._default_config()
        
        self.config = config['simulation']
        self.n_subcarriers = int(self.config['n_subcarriers'])
        self.sample_rate = float(self.config['sample_rate'])
        self.carrier_freq = float(self.config['carrier_freq'])
        self.bandwidth = float(self.config['bandwidth'])
        
        # Speed of light
        self.c = 3e8
        
        # Subcarrier frequencies
        self.subcarrier_freqs = self._compute_subcarrier_frequencies()
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'simulation': {
                'n_subcarriers': 30,
                'sample_rate': 100,
                'carrier_freq': 5.2e9,
                'bandwidth': 20e6,
                'room': {
                    'dimensions': [5, 4, 3],
                    'tx_position': [0, 2, 1.5],
                    'rx_position': [5, 2, 1.5]
                },
                'multipath': {
                    'n_paths': 4,
                    'enable_los': True,
                    'reflection_coeff': 0.7
                },
                'noise': {
                    'thermal_floor': -90,
                    'snr_db': 20,
                    'interference_prob': 0.1,
                    'interference_power': -70
                },
                'human': {
                    'rcs': 0.5,
                    'height': 1.7,
                    'width': 0.4
                }
            }
        }
    
    def _compute_subcarrier_frequencies(self) -> np.ndarray:
        """
        Compute OFDM subcarrier frequencies.
        
        Returns:
            Array of subcarrier frequencies in Hz
        """
        subcarrier_spacing = self.bandwidth / self.n_subcarriers
        # Center frequencies around carrier
        indices = np.arange(self.n_subcarriers) - self.n_subcarriers // 2
        return self.carrier_freq + indices * subcarrier_spacing
    
    def generate_empty_room(
        self,
        duration: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate CSI for an empty room (static environment).
        
        Args:
            duration: Duration in seconds
            seed: Random seed for reproducibility
            
        Returns:
            CSI matrix of shape (n_samples, n_subcarriers) - complex values
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = int(duration * self.sample_rate)
        
        # Static channel with small variations
        csi = np.zeros((n_samples, self.n_subcarriers), dtype=complex)
        
        # Generate static multipath components
        paths = self._generate_static_paths()
        
        for t in range(n_samples):
            for f_idx, freq in enumerate(self.subcarrier_freqs):
                # Sum contributions from all paths
                h = 0
                for path in paths:
                    amplitude = path['amplitude']
                    delay = path['delay']
                    phase_shift = -2 * np.pi * freq * delay
                    h += amplitude * np.exp(1j * phase_shift)
                
                csi[t, f_idx] = h
        
        # Add noise
        csi = self._add_noise(csi)
        
        # Add small temporal variations (environmental changes)
        csi = self._add_temporal_drift(csi, drift_scale=0.02)
        
        return csi
    
    def generate_occupied_room(
        self,
        duration: float,
        activity: str = "standing",
        human_position: Optional[List[float]] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate CSI for an occupied room with human presence.
        
        Args:
            duration: Duration in seconds
            activity: Type of activity ("standing", "walking", "sitting")
            human_position: Initial [x, y, z] position. If None, uses room center.
            seed: Random seed for reproducibility
            
        Returns:
            CSI matrix of shape (n_samples, n_subcarriers) - complex values
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = int(duration * self.sample_rate)
        
        # Get room dimensions
        room_dims = self.config['room']['dimensions']
        if human_position is None:
            human_position = [room_dims[0]/2, room_dims[1]/2, 1.0]
        
        # Generate human motion trajectory
        trajectory = self._generate_trajectory(
            human_position, 
            n_samples, 
            activity
        )
        
        csi = np.zeros((n_samples, self.n_subcarriers), dtype=complex)
        
        # Generate base static paths
        static_paths = self._generate_static_paths()
        
        tx_pos = np.array(self.config['room']['tx_position'])
        rx_pos = np.array(self.config['room']['rx_position'])
        
        for t in range(n_samples):
            human_pos = trajectory[t]
            
            # Generate dynamic path through human
            human_path = self._compute_human_path(human_pos, tx_pos, rx_pos, t)
            
            # Combine static and dynamic paths
            all_paths = static_paths + [human_path]
            
            for f_idx, freq in enumerate(self.subcarrier_freqs):
                h = 0
                for path in all_paths:
                    amplitude = path['amplitude']
                    delay = path['delay']
                    doppler = path.get('doppler', 0)
                    
                    # Phase shift due to delay and Doppler
                    phase_shift = -2 * np.pi * freq * delay
                    doppler_shift = 2 * np.pi * doppler * t / self.sample_rate
                    
                    h += amplitude * np.exp(1j * (phase_shift + doppler_shift))
                
                csi[t, f_idx] = h
        
        # Add noise
        csi = self._add_noise(csi)
        
        return csi
    
    def _generate_static_paths(self) -> List[Dict]:
        """
        Generate static multipath components (walls, ceiling, floor).
        
        Returns:
            List of path dictionaries with amplitude and delay
        """
        paths = []
        n_paths = self.config['multipath']['n_paths']
        enable_los = self.config['multipath']['enable_los']
        reflection_coeff = self.config['multipath']['reflection_coeff']
        
        tx_pos = np.array(self.config['room']['tx_position'])
        rx_pos = np.array(self.config['room']['rx_position'])
        
        # Line-of-sight path
        if enable_los:
            los_distance = np.linalg.norm(rx_pos - tx_pos)
            los_delay = los_distance / self.c
            los_amplitude = 1.0 / los_distance  # Free space path loss
            
            paths.append({
                'amplitude': los_amplitude,
                'delay': los_delay,
                'type': 'los'
            })
        
        # Reflected paths (simplified - random reflections)
        for i in range(n_paths - 1):
            # Reflected path is longer
            extra_distance = np.random.uniform(0.5, 3.0)
            distance = np.linalg.norm(rx_pos - tx_pos) + extra_distance
            delay = distance / self.c
            
            # Amplitude decreases with reflections
            n_reflections = np.random.randint(1, 3)
            amplitude = (reflection_coeff ** n_reflections) / distance
            
            paths.append({
                'amplitude': amplitude,
                'delay': delay,
                'type': 'reflected'
            })
        
        return paths
    
    def _compute_human_path(
        self, 
        human_pos: np.ndarray, 
        tx_pos: np.ndarray, 
        rx_pos: np.ndarray,
        time_idx: int
    ) -> Dict:
        """
        Compute path component that passes through human body.
        
        Args:
            human_pos: Human position [x, y, z]
            tx_pos: Transmitter position
            rx_pos: Receiver position
            time_idx: Current time index
            
        Returns:
            Path dictionary with amplitude, delay, and Doppler
        """
        # Distance: TX -> Human -> RX
        d1 = np.linalg.norm(human_pos - tx_pos)
        d2 = np.linalg.norm(rx_pos - human_pos)
        total_distance = d1 + d2
        
        delay = total_distance / self.c
        
        # Amplitude based on radar cross-section and distance
        rcs = self.config['human']['rcs']
        amplitude = np.sqrt(rcs) / (d1 * d2)
        
        # Estimate Doppler shift (simplified)
        # In practice, would compute from velocity component along path
        # For now, add small random Doppler to simulate micro-movements
        doppler = np.random.normal(0, 2)  # Hz, small for standing/sitting
        
        return {
            'amplitude': amplitude,
            'delay': delay,
            'doppler': doppler,
            'type': 'human'
        }
    
    def _generate_trajectory(
        self,
        start_pos: List[float],
        n_samples: int,
        activity: str
    ) -> np.ndarray:
        """
        Generate human motion trajectory based on activity.
        
        Args:
            start_pos: Starting position [x, y, z]
            n_samples: Number of time samples
            activity: Activity type
            
        Returns:
            Trajectory array of shape (n_samples, 3)
        """
        trajectory = np.zeros((n_samples, 3))
        trajectory[0] = start_pos
        
        if activity == "standing":
            # Small random movements (breathing, swaying)
            for t in range(1, n_samples):
                trajectory[t] = trajectory[t-1] + np.random.normal(0, 0.005, 3)
        
        elif activity == "sitting":
            # Even smaller movements
            for t in range(1, n_samples):
                trajectory[t] = trajectory[t-1] + np.random.normal(0, 0.002, 3)
        
        elif activity == "walking":
            # Linear motion with periodic component (steps)
            room_dims = self.config['room']['dimensions']
            walking_speed = 1.0  # m/s
            step_freq = 2.0  # Hz (2 steps per second)
            
            for t in range(1, n_samples):
                time = t / self.sample_rate
                
                # Linear motion
                x_vel = walking_speed
                trajectory[t, 0] = start_pos[0] + x_vel * time
                
                # Periodic vertical motion (bouncing while walking)
                trajectory[t, 2] = start_pos[2] + 0.02 * np.sin(2 * np.pi * step_freq * time)
                
                # Keep y constant
                trajectory[t, 1] = start_pos[1]
                
                # Bounce off walls
                if trajectory[t, 0] > room_dims[0]:
                    trajectory[t, 0] = room_dims[0] - (trajectory[t, 0] - room_dims[0])
                elif trajectory[t, 0] < 0:
                    trajectory[t, 0] = -trajectory[t, 0]
        
        else:
            # Default: stationary
            trajectory[:] = start_pos
        
        return trajectory
    
    def _add_noise(self, csi: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to CSI measurements.
        
        Args:
            csi: Clean CSI matrix
            
        Returns:
            Noisy CSI matrix
        """
        # Thermal noise (Gaussian)
        snr_db = self.config['noise']['snr_db']
        signal_power = np.mean(np.abs(csi) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*csi.shape) + 1j * np.random.randn(*csi.shape)
        )
        
        csi_noisy = csi + noise
        
        # Interference bursts (random)
        interference_prob = self.config['noise']['interference_prob']
        interference_mask = np.random.rand(csi.shape[0], 1) < interference_prob
        interference_power_db = self.config['noise']['interference_power']
        interference_power = 10 ** (interference_power_db / 10)
        
        interference = np.sqrt(interference_power / 2) * (
            np.random.randn(*csi.shape) + 1j * np.random.randn(*csi.shape)
        )
        
        csi_noisy += interference * interference_mask
        
        return csi_noisy
    
    def _add_temporal_drift(self, csi: np.ndarray, drift_scale: float = 0.01) -> np.ndarray:
        """
        Add slow temporal drift to simulate environmental changes.
        
        Args:
            csi: CSI matrix
            drift_scale: Scale of drift
            
        Returns:
            CSI with temporal drift
        """
        n_samples = csi.shape[0]
        
        # Low-frequency drift
        drift_freq = 0.1  # Hz
        t = np.arange(n_samples) / self.sample_rate
        
        drift_amplitude = drift_scale * np.exp(1j * 2 * np.pi * drift_freq * t)
        drift_amplitude = drift_amplitude[:, np.newaxis]
        
        return csi * drift_amplitude
    
    def generate_batch(
        self,
        n_empty: int,
        n_occupied: int,
        duration: float,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of CSI samples for training.
        
        Args:
            n_empty: Number of empty room samples
            n_occupied: Number of occupied room samples
            duration: Duration of each sample in seconds
            seed: Random seed
            
        Returns:
            Tuple of (csi_data, labels)
            - csi_data: shape (n_samples, n_timesteps, n_subcarriers)
            - labels: shape (n_samples,) - 0 for empty, 1 for occupied
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = n_empty + n_occupied
        n_timesteps = int(duration * self.sample_rate)
        
        csi_data = np.zeros((n_samples, n_timesteps, self.n_subcarriers), dtype=complex)
        labels = np.zeros(n_samples, dtype=int)
        
        # Generate empty room samples
        for i in range(n_empty):
            csi_data[i] = self.generate_empty_room(duration, seed=seed+i if seed else None)
            labels[i] = 0
        
        # Generate occupied room samples
        for i in range(n_occupied):
            idx = n_empty + i
            activity = np.random.choice(["standing", "sitting", "walking"])
            csi_data[idx] = self.generate_occupied_room(
                duration, 
                activity=activity, 
                seed=seed+idx if seed else None
            )
            labels[idx] = 1
        
        return csi_data, labels


if __name__ == "__main__":
    # Quick test
    generator = CSIGenerator()
    
    print("Generating empty room CSI...")
    csi_empty = generator.generate_empty_room(duration=2.0, seed=42)
    print(f"Empty room CSI shape: {csi_empty.shape}")
    print(f"Mean amplitude: {np.mean(np.abs(csi_empty)):.4f}")
    
    print("\nGenerating occupied room CSI...")
    csi_occupied = generator.generate_occupied_room(duration=2.0, activity="walking", seed=42)
    print(f"Occupied room CSI shape: {csi_occupied.shape}")
    print(f"Mean amplitude: {np.mean(np.abs(csi_occupied)):.4f}")
    
    print("\nGenerating batch...")
    csi_batch, labels = generator.generate_batch(n_empty=10, n_occupied=10, duration=5.0, seed=42)
    print(f"Batch shape: {csi_batch.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Empty samples: {np.sum(labels == 0)}, Occupied samples: {np.sum(labels == 1)}")
    
    print("\n✅ CSI Generator working correctly!")

