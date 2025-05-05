#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Device Anomaly Detection Example

This script demonstrates how to use the Multi-Modal Bio-Signal System
to detect anomalies across multiple wireless devices using the transformer-based
anomaly detection module.

The example simulates data from different bio-signal sources (ECG, EMG, PPG)
and shows how to analyze them together using the system's integrated
transformer analysis module.

Based on the patent "Multi-Modal Bio-Stimulation, Diagnosis and Treatment System
Using Multiple Wireless Devices and Method for the Same"
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from central_system.transformer_analysis import TransformerAnalysisModule
from central_system.config import SystemConfig
from central_system.bluetooth_manager import BluetoothManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_device_example")

class MultiDeviceExample:
    """
    Example class for demonstrating multi-device anomaly detection
    """
    
    def __init__(self, output_dir="example_output", simulation_mode=True):
        """
        Initialize the example
        
        Args:
            output_dir: Directory to save output files
            simulation_mode: Whether to use simulated devices
        """
        self.output_dir = output_dir
        self.simulation_mode = simulation_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize system components
        self.config = SystemConfig()
        self.config.simulation_mode = simulation_mode
        
        # Initialize transformer analysis module
        self.transformer_analyzer = TransformerAnalysisModule(
            model_dir=os.path.join(output_dir, "models"),
            device="cpu"  # Use CPU for example (change to "cuda" for GPU)
        )
        
        # Initialize device connection
        if not simulation_mode:
            self.bluetooth_manager = BluetoothManager(self.config.bluetooth_config)
        
        # Setup signal parameters
        self.setup_signal_parameters()
        
        # Initialize buffers for collected data
        self.collected_data = {
            "ecg": np.zeros((0, 3)),  # ECG data: [raw, filtered, derivative]
            "emg": np.zeros((0, 4)),  # EMG data: [raw, filtered, rms, frequency]
            "ppg": np.zeros((0, 3))   # PPG data: [raw, filtered, pulse_rate]
        }
        
        # Anomaly injection settings
        self.anomaly_settings = {
            "ecg": {"enabled": True, "start": 10.0, "duration": 5.0, "amplitude": 0.5},
            "emg": {"enabled": True, "start": 15.0, "duration": 3.0, "amplitude": 0.8},
            "ppg": {"enabled": True, "start": 18.0, "duration": 2.0, "amplitude": 0.4}
        }
        
        # TENS stimulation parameters
        self.stimulation_params = {
            "frequency": 50.0,  # Hz
            "pulse_width": 200.0,  # µs
            "intensity": 20.0,  # mA
            "mode": "continuous",
            "duration": 30.0,  # seconds
            "ramp_up": 2.0,  # seconds
            "ramp_down": 2.0,  # seconds
            "max_intensity": 50.0,  # mA
            "min_intensity": 5.0    # mA
        }
        
        logger.info("Multi-device example initialized")
    
    def setup_signal_parameters(self):
        """
        Setup parameters for signal generation
        """
        # ECG signal parameters
        self.ecg_params = {
            "sampling_rate": 200,  # Hz
            "heart_rate": 60,  # BPM
            "heart_rate_variability": 5.0,  # BPM
            "noise_level": 0.05
        }
        
        # EMG signal parameters
        self.emg_params = {
            "sampling_rate": 200,  # Hz
            "base_frequency": 80,  # Hz
            "amplitude": 0.5,
            "burst_probability": 0.2,
            "noise_level": 0.1
        }
        
        # PPG signal parameters
        self.ppg_params = {
            "sampling_rate": 50,  # Hz
            "heart_rate": 65,  # BPM (slightly different from ECG for realism)
            "amplitude": 0.8,
            "noise_level": 0.03
        }
    
    def generate_ecg_data(self, duration, current_time=0.0):
        """
        Generate ECG signal data
        
        Args:
            duration: Duration of signal in seconds
            current_time: Current simulation time
            
        Returns:
            ECG data array with shape [samples, 3]
        """
        # Calculate number of samples
        num_samples = int(duration * self.ecg_params["sampling_rate"])
        
        # Create time array
        t = np.linspace(current_time, current_time + duration, num_samples)
        
        # Calculate varying heart rate
        heart_rate = self.ecg_params["heart_rate"] + \
                    self.ecg_params["heart_rate_variability"] * np.sin(0.1 * t)
        
        # Calculate beat interval in seconds
        beat_interval = 60.0 / heart_rate
        
        # Generate raw ECG signal
        ecg_raw = np.zeros(num_samples)
        
        for i, ti in enumerate(t):
            # Calculate phase within heartbeat
            phase = (ti % beat_interval[i]) / beat_interval[i] * 2 * np.pi
            
            # Different components of ECG signal
            if phase < 0.1 * 2 * np.pi:
                # P wave
                ecg_raw[i] = 0.2 * np.sin(phase / 0.1)
            elif phase < 0.2 * 2 * np.pi:
                # PR segment
                ecg_raw[i] = 0
            elif phase < 0.3 * 2 * np.pi:
                # QRS complex
                ecg_raw[i] = -0.5 * np.sin(phase / 0.05) + 1.5 * np.sin((phase - 0.25) / 0.05)
            elif phase < 0.7 * 2 * np.pi:
                # ST segment
                ecg_raw[i] = 0
            elif phase < 0.9 * 2 * np.pi:
                # T wave
                ecg_raw[i] = 0.3 * np.sin((phase - 0.7) / 0.2)
            else:
                ecg_raw[i] = 0
        
        # Add noise
        ecg_raw += self.ecg_params["noise_level"] * np.random.randn(num_samples)
        
        # Inject anomaly if needed
        if self.anomaly_settings["ecg"]["enabled"]:
            anomaly_start = self.anomaly_settings["ecg"]["start"]
            anomaly_end = anomaly_start + self.anomaly_settings["ecg"]["duration"]
            
            if current_time <= anomaly_end and current_time + duration >= anomaly_start:
                # Calculate indices for anomaly
                start_idx = max(0, int((anomaly_start - current_time) * self.ecg_params["sampling_rate"]))
                end_idx = min(num_samples, int((anomaly_end - current_time) * self.ecg_params["sampling_rate"]))
                
                if start_idx < end_idx:
                    # Add arrhythmia-like pattern
                    anomaly_amplitude = self.anomaly_settings["ecg"]["amplitude"]
                    anomaly_length = end_idx - start_idx
                    
                    # Create irregular pattern (premature ventricular contraction)
                    anomaly = np.zeros(anomaly_length)
                    
                    # Add PVC-like shape
                    anomaly_t = np.linspace(0, 2*np.pi, anomaly_length)
                    anomaly = -1.5 * anomaly_amplitude * np.sin(anomaly_t) + \
                              anomaly_amplitude * np.sin(3 * anomaly_t)
                    
                    # Apply anomaly
                    ecg_raw[start_idx:end_idx] += anomaly
        
        # Calculate filtered version (simple moving average for demonstration)
        window_size = 5
        ecg_filtered = np.zeros_like(ecg_raw)
        for i in range(num_samples):
            start = max(0, i - window_size // 2)
            end = min(num_samples, i + window_size // 2 + 1)
            ecg_filtered[i] = np.mean(ecg_raw[start:end])
        
        # Calculate derivative
        ecg_derivative = np.gradient(ecg_filtered)
        
        # Combine into multi-channel data
        ecg_data = np.column_stack((ecg_raw, ecg_filtered, ecg_derivative))
        
        return ecg_data
    
    def generate_emg_data(self, duration, current_time=0.0):
        """
        Generate EMG signal data
        
        Args:
            duration: Duration of signal in seconds
            current_time: Current simulation time
            
        Returns:
            EMG data array with shape [samples, 4]
        """
        # Calculate number of samples
        num_samples = int(duration * self.emg_params["sampling_rate"])
        
        # Create time array
        t = np.linspace(current_time, current_time + duration, num_samples)
        
        # Generate raw EMG signal (base + bursts)
        emg_raw = np.zeros(num_samples)
        
        # Base signal (low-amplitude noise)
        base_signal = self.emg_params["noise_level"] * np.random.randn(num_samples)
        
        # Add bursts of activity
        for i in range(0, num_samples, 200):  # Check every second
            if np.random.random() < self.emg_params["burst_probability"]:
                # Create a burst
                burst_length = np.random.randint(20, 100)  # 0.1-0.5 seconds
                if i + burst_length > num_samples:
                    burst_length = num_samples - i
                
                # Burst is amplitude-modulated noise
                envelope = self.emg_params["amplitude"] * np.sin(np.linspace(0, np.pi, burst_length))**2
                burst = envelope * np.random.randn(burst_length)
                
                # Add burst to signal
                emg_raw[i:i+burst_length] += burst
        
        # Add base signal
        emg_raw += base_signal
        
        # Inject anomaly if needed
        if self.anomaly_settings["emg"]["enabled"]:
            anomaly_start = self.anomaly_settings["emg"]["start"]
            anomaly_end = anomaly_start + self.anomaly_settings["emg"]["duration"]
            
            if current_time <= anomaly_end and current_time + duration >= anomaly_start:
                # Calculate indices for anomaly
                start_idx = max(0, int((anomaly_start - current_time) * self.emg_params["sampling_rate"]))
                end_idx = min(num_samples, int((anomaly_end - current_time) * self.emg_params["sampling_rate"]))
                
                if start_idx < end_idx:
                    # Add sustained contraction anomaly
                    anomaly_amplitude = self.anomaly_settings["emg"]["amplitude"]
                    anomaly_length = end_idx - start_idx
                    
                    # Create sustained high-amplitude pattern
                    anomaly = anomaly_amplitude * (0.8 + 0.4 * np.random.rand(anomaly_length))
                    
                    # Apply anomaly
                    emg_raw[start_idx:end_idx] += anomaly
        
        # Calculate filtered version
        window_size = 9
        emg_filtered = np.zeros_like(emg_raw)
        for i in range(num_samples):
            start = max(0, i - window_size // 2)
            end = min(num_samples, i + window_size // 2 + 1)
            emg_filtered[i] = np.mean(emg_raw[start:end])
        
        # Calculate RMS in sliding windows
        window_size = 40  # 200ms windows
        emg_rms = np.zeros_like(emg_raw)
        for i in range(num_samples):
            start = max(0, i - window_size // 2)
            end = min(num_samples, i + window_size // 2 + 1)
            emg_rms[i] = np.sqrt(np.mean(emg_raw[start:end]**2))
        
        # Calculate frequency content (simplified)
        emg_frequency = np.zeros_like(emg_raw)
        for i in range(0, num_samples, window_size):
            end = min(i + window_size, num_samples)
            if end - i > 10:  # Need enough samples for FFT
                segment = emg_raw[i:end]
                fft = np.abs(np.fft.rfft(segment))
                # Use simple metric (normalized mean frequency)
                freqs = np.fft.rfftfreq(len(segment), 1/self.emg_params["sampling_rate"])
                if np.sum(fft) > 0:
                    mean_freq = np.sum(freqs * fft) / np.sum(fft) / 100  # Normalize
                    emg_frequency[i:end] = mean_freq
        
        # Combine into multi-channel data
        emg_data = np.column_stack((emg_raw, emg_filtered, emg_rms, emg_frequency))
        
        return emg_data
    
    def generate_ppg_data(self, duration, current_time=0.0):
        """
        Generate PPG signal data
        
        Args:
            duration: Duration of signal in seconds
            current_time: Current simulation time
            
        Returns:
            PPG data array with shape [samples, 3]
        """
        # Calculate number of samples
        num_samples = int(duration * self.ppg_params["sampling_rate"])
        
        # Create time array
        t = np.linspace(current_time, current_time + duration, num_samples)
        
        # Calculate heart rate (slightly different from ECG for realism)
        heart_rate = self.ppg_params["heart_rate"] + \
                    3.0 * np.sin(0.1 * t)  # Some variability
        
        # Calculate beat interval in seconds
        beat_interval = 60.0 / heart_rate
        
        # Generate raw PPG signal
        ppg_raw = np.zeros(num_samples)
        
        for i, ti in enumerate(t):
            # Calculate phase within heartbeat
            phase = (ti % beat_interval[i]) / beat_interval[i] * 2 * np.pi
            
            # PPG waveform (systolic rise, diastolic decay)
            # Modified Gaussian function
            if phase < np.pi:
                # Systolic rise
                ppg_raw[i] = self.ppg_params["amplitude"] * \
                            np.exp(-((phase - 0.5*np.pi) / 0.4)**2)
            else:
                # Diastolic decay
                ppg_raw[i] = self.ppg_params["amplitude"] * 0.3 * \
                            np.exp(-((phase - 0.5*np.pi) / 1.2)**2)
        
        # Add baseline drift (respiration effect)
        baseline_drift = 0.1 * np.sin(0.05 * t * 2 * np.pi)
        ppg_raw += baseline_drift
        
        # Add noise
        ppg_raw += self.ppg_params["noise_level"] * np.random.randn(num_samples)
        
        # Inject anomaly if needed
        if self.anomaly_settings["ppg"]["enabled"]:
            anomaly_start = self.anomaly_settings["ppg"]["start"]
            anomaly_end = anomaly_start + self.anomaly_settings["ppg"]["duration"]
            
            if current_time <= anomaly_end and current_time + duration >= anomaly_start:
                # Calculate indices for anomaly
                start_idx = max(0, int((anomaly_start - current_time) * self.ppg_params["sampling_rate"]))
                end_idx = min(num_samples, int((anomaly_end - current_time) * self.ppg_params["sampling_rate"]))
                
                if start_idx < end_idx:
                    # Add poor perfusion anomaly (reduced amplitude)
                    anomaly_amplitude = self.anomaly_settings["ppg"]["amplitude"]
                    anomaly_length = end_idx - start_idx
                    
                    # Create damped pattern
                    damping = 1.0 - anomaly_amplitude * np.ones(anomaly_length)
                    
                    # Apply anomaly (dampen signal)
                    ppg_raw[start_idx:end_idx] *= damping
        
        # Calculate filtered version
        window_size = 5
        ppg_filtered = np.zeros_like(ppg_raw)
        for i in range(num_samples):
            start = max(0, i - window_size // 2)
            end = min(num_samples, i + window_size // 2 + 1)
            ppg_filtered[i] = np.mean(ppg_raw[start:end])
        
        # Calculate pulse rate
        pulse_rate = np.zeros_like(ppg_raw)
        for i in range(num_samples):
            # Use heart rate with slight delay
            idx = max(0, min(num_samples-1, i - 10))
            pulse_rate[i] = heart_rate[idx] / 200.0  # Normalize to 0-1 range (assuming max HR 200)
        
        # Combine into multi-channel data
        ppg_data = np.column_stack((ppg_raw, ppg_filtered, pulse_rate))
        
        return ppg_data
    
    def run_simulation(self, duration=30.0, chunk_duration=0.5):
        """
        Run simulation for specified duration
        
        Args:
            duration: Total duration in seconds
            chunk_duration: Duration of each data chunk in seconds
        """
        logger.info(f"Starting simulation for {duration} seconds")
        
        # Clear collected data
        self.collected_data = {
            "ecg": np.zeros((0, 3)),
            "emg": np.zeros((0, 4)),
            "ppg": np.zeros((0, 3))
        }
        
        # Loop through time chunks
        current_time = 0.0
        while current_time < duration:
            # Calculate actual chunk duration (might be less at the end)
            actual_chunk_duration = min(chunk_duration, duration - current_time)
            if actual_chunk_duration <= 0:
                break
            
            # Generate data from each device
            ecg_chunk = self.generate_ecg_data(actual_chunk_duration, current_time)
            emg_chunk = self.generate_emg_data(actual_chunk_duration, current_time)
            ppg_chunk = self.generate_ppg_data(actual_chunk_duration, current_time)
            
            # Add to collected data
            self.collected_data["ecg"] = np.vstack((self.collected_data["ecg"], ecg_chunk))
            self.collected_data["emg"] = np.vstack((self.collected_data["emg"], emg_chunk))
            self.collected_data["ppg"] = np.vstack((self.collected_data["ppg"], ppg_chunk))
            
            # Feed data to transformer analyzer
            self.transformer_analyzer.add_data("ecg", ecg_chunk, timestamp=current_time)
            self.transformer_analyzer.add_data("emg", emg_chunk, timestamp=current_time)
            self.transformer_analyzer.add_data("ppg", ppg_chunk, timestamp=current_time)
            
            # Analyze signals
            ecg_result = self.transformer_analyzer.analyze_signal("ecg")
            emg_result = self.transformer_analyzer.analyze_signal("emg")
            ppg_result = self.transformer_analyzer.analyze_signal("ppg")
            
            # Log analysis results
            if ecg_result:
                if ecg_result.get("anomaly_detected", False):
                    logger.info(f"Time {current_time:.1f}s: ECG anomaly detected! Score: {ecg_result.get('anomaly_score', 0):.4f}")
                    
                    # Adjust stimulation parameters based on anomaly
                    adjusted_params = self.transformer_analyzer.get_stimulation_parameters(
                        "ecg", self.stimulation_params
                    )
                    
                    # Log parameter adjustments
                    logger.info(f"  - Adjusted TENS frequency: {adjusted_params['frequency']:.1f} Hz")
                    logger.info(f"  - Adjusted TENS intensity: {adjusted_params['intensity']:.1f} mA")
            
            if emg_result:
                if emg_result.get("anomaly_detected", False):
                    logger.info(f"Time {current_time:.1f}s: EMG anomaly detected! Score: {emg_result.get('anomaly_score', 0):.4f}")
                    
                    # Adjust stimulation parameters based on anomaly
                    adjusted_params = self.transformer_analyzer.get_stimulation_parameters(
                        "emg", self.stimulation_params
                    )
                    
                    # Log parameter adjustments
                    logger.info(f"  - Adjusted TENS pulse width: {adjusted_params['pulse_width']:.1f} µs")
            
            if ppg_result:
                if ppg_result.get("anomaly_detected", False):
                    logger.info(f"Time {current_time:.1f}s: PPG anomaly detected! Score: {ppg_result.get('anomaly_score', 0):.4f}")
                    
                    # Adjust stimulation parameters based on anomaly
                    adjusted_params = self.transformer_analyzer.get_stimulation_parameters(
                        "ppg", self.stimulation_params
                    )
                    
                    # Log parameter adjustments
                    if "mode" in adjusted_params:
                        logger.info(f"  - Adjusted TENS mode: {adjusted_params['mode']}")
            
            # Increment time
            current_time += actual_chunk_duration
            
            # Small delay for real-time simulation
            time.sleep(actual_chunk_duration * 0.1)  # 10x speedup
        
        logger.info("Simulation completed")
        
        # Generate final report
        self._generate_report()
    
    def _generate_report(self):
        """
        Generate analysis report and visualizations
        """
        logger.info("Generating report")
        
        # Get time period covered by data
        duration = self.collected_data["ecg"].shape[0] / self.ecg_params["sampling_rate"]
        time_period = (0.0, duration)
        
        # Generate comprehensive report
        report = self.transformer_analyzer.generate_report(
            signal_data=self.collected_data,
            time_period=time_period,
            save_path=os.path.join(self.output_dir, "report")
        )
        
        # Log report recommendations
        logger.info("Report recommendations:")
        for key, value in report['recommendations'].items():
            logger.info(f"  - {key}: {value}")
        
        # Generate additional visualization
        self._visualize_signals()
        
        logger.info(f"Report saved to {self.output_dir}")
    
    def _visualize_signals(self):
        """
        Generate visualization of all signals
        """
        # Create time axes
        ecg_time = np.linspace(0, self.collected_data["ecg"].shape[0] / self.ecg_params["sampling_rate"], 
                              self.collected_data["ecg"].shape[0])
        
        emg_time = np.linspace(0, self.collected_data["emg"].shape[0] / self.emg_params["sampling_rate"],
                              self.collected_data["emg"].shape[0])
        
        ppg_time = np.linspace(0, self.collected_data["ppg"].shape[0] / self.ppg_params["sampling_rate"],
                              self.collected_data["ppg"].shape[0])
        
        # Create figure
        plt.figure(figsize=(10, 12))
        
        # Plot ECG signal
        plt.subplot(3, 1, 1)
        plt.plot(ecg_time, self.collected_data["ecg"][:, 0], 'b-', label='Raw')
        plt.plot(ecg_time, self.collected_data["ecg"][:, 1], 'g-', label='Filtered')
        
        # Highlight ECG anomaly region
        if self.anomaly_settings["ecg"]["enabled"]:
            plt.axvspan(self.anomaly_settings["ecg"]["start"], 
                       self.anomaly_settings["ecg"]["start"] + self.anomaly_settings["ecg"]["duration"],
                       alpha=0.2, color='red')
            
        plt.title('ECG Signal')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot EMG signal
        plt.subplot(3, 1, 2)
        plt.plot(emg_time, self.collected_data["emg"][:, 0], 'b-', label='Raw')
        plt.plot(emg_time, self.collected_data["emg"][:, 2], 'r-', label='RMS')
        
        # Highlight EMG anomaly region
        if self.anomaly_settings["emg"]["enabled"]:
            plt.axvspan(self.anomaly_settings["emg"]["start"], 
                       self.anomaly_settings["emg"]["start"] + self.anomaly_settings["emg"]["duration"],
                       alpha=0.2, color='red')
            
        plt.title('EMG Signal')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot PPG signal
        plt.subplot(3, 1, 3)
        plt.plot(ppg_time, self.collected_data["ppg"][:, 0], 'b-', label='Raw')
        plt.plot(ppg_time, self.collected_data["ppg"][:, 1], 'g-', label='Filtered')
        
        # Highlight PPG anomaly region
        if self.anomaly_settings["ppg"]["enabled"]:
            plt.axvspan(self.anomaly_settings["ppg"]["start"], 
                       self.anomaly_settings["ppg"]["start"] + self.anomaly_settings["ppg"]["duration"],
                       alpha=0.2, color='red')
            
        plt.title('PPG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "multi_device_signals.png"))
        plt.close()


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Multi-Device Anomaly Detection Example')
    
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Simulation duration in seconds')
    
    parser.add_argument('--output', type=str, default='example_output',
                        help='Output directory for results')
    
    parser.add_argument('--real', action='store_true',
                        help='Use real devices instead of simulation')
    
    return parser.parse_args()


def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_args()
    
    # Create example
    example = MultiDeviceExample(
        output_dir=args.output,
        simulation_mode=not args.real
    )
    
    # Run simulation
    example.run_simulation(duration=args.duration)


if __name__ == "__main__":
    main()
