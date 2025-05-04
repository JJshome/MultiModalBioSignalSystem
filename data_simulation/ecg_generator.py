import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import random
import time
import os
import json

class ECGGenerator:
    """Generate realistic ECG signals for simulation and testing"""
    
    def __init__(self, 
                sampling_rate: int = 250, 
                duration_seconds: int = 60, 
                heart_rate: float = 70,
                hrv_level: float = 50,  # ms, SDNN
                noise_level: float = 0.05,
                arrhythmia_probability: float = 0.0):
        
        self.sampling_rate = sampling_rate
        self.duration_seconds = duration_seconds
        self.heart_rate = heart_rate
        self.hrv_level = hrv_level
        self.noise_level = noise_level
        self.arrhythmia_probability = arrhythmia_probability
        
        # Create ECG template
        self.ecg_template = self._create_ecg_template()
        
        # Simulation data
        self.timestamps = None
        self.ecg_signal = None
        self.r_peaks = None
        self.rr_intervals = None
        
    def _create_ecg_template(self, template_size: int = 100) -> np.ndarray:
        """Create a template for ECG waveform"""
        # Time vector
        t = np.linspace(0, 2*np.pi, template_size)
        
        # P wave
        p_wave = 0.25 * np.exp(-((t - 0.4)**2) / 0.05)
        
        # QRS complex
        q_wave = -0.1 * np.exp(-((t - 1.0)**2) / 0.01)
        r_wave = 1.0 * np.exp(-((t - 1.2)**2) / 0.01)
        s_wave = -0.3 * np.exp(-((t - 1.4)**2) / 0.01)
        qrs_complex = q_wave + r_wave + s_wave
        
        # T wave
        t_wave = 0.35 * np.exp(-((t - 2.0)**2) / 0.1)
        
        # Complete ECG cycle
        ecg_template = p_wave + qrs_complex + t_wave
        
        # Normalize
        ecg_template = ecg_template / np.max(np.abs(ecg_template))
        
        return ecg_template
    
    def generate_ecg(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a full ECG signal with specified parameters"""
        # Create time vector
        num_samples = self.duration_seconds * self.sampling_rate
        timestamps = np.linspace(0, self.duration_seconds, num_samples)
        
        # Create empty signal
        ecg_signal = np.zeros(num_samples)
        
        # Generate R-peak times with heart rate and HRV variability
        r_peak_times = self._generate_heartbeats(self.heart_rate, self.hrv_level, self.duration_seconds)
        r_peaks = []
        
        # Add QRS complexes at each R-peak time
        template_len = len(self.ecg_template)
        for r_time in r_peak_times:
            # Find index of R peak
            r_idx = int(r_time * self.sampling_rate)
            
            # Avoid exceeding array bounds
            if r_idx >= num_samples - template_len:
                continue
            
            # Record R peak
            r_peaks.append(r_idx)
            
            # Add template to signal
            offset = r_idx - template_len // 4  # Offset to place R peak at correct position
            for i in range(template_len):
                if 0 <= offset + i < num_samples:
                    ecg_signal[offset + i] += self.ecg_template[i]
        
        # Add noise
        noise = self.noise_level * np.random.randn(num_samples)
        ecg_signal += noise
        
        # Add baseline wander (respiratory effect)
        baseline_wander = 0.2 * np.sin(2 * np.pi * 0.2 * timestamps)  # ~12 breaths per minute
        ecg_signal += baseline_wander
        
        # Add arrhythmias if specified
        if self.arrhythmia_probability > 0:
            ecg_signal = self._add_arrhythmias(ecg_signal, r_peaks)
        
        # Store simulated data
        self.timestamps = timestamps
        self.ecg_signal = ecg_signal
        self.r_peaks = np.array(r_peaks)
        self.rr_intervals = np.diff(r_peak_times) * 1000  # in milliseconds
        
        return timestamps, ecg_signal
    
    def _generate_heartbeats(self, heart_rate: float, hrv_level: float, duration: float) -> List[float]:
        """Generate R-peak times with realistic heart rate and HRV"""
        # Calculate initial RR interval in seconds
        mean_rri = 60.0 / heart_rate
        
        # Standard deviation in seconds
        std_rri = hrv_level / 1000.0
        
        # Generate first peak
        r_peak_times = [0]
        current_time = 0
        
        # Generate subsequent peaks
        while current_time < duration:
            # Calculate next RR interval with variation
            if random.random() < self.arrhythmia_probability:
                # Occasionally add arrhythmia (premature beat or skipped beat)
                if random.random() < 0.5:
                    # Premature beat
                    rri = mean_rri * 0.7
                else:
                    # Skipped beat
                    rri = mean_rri * 1.5
            else:
                # Normal beat with HRV
                rri = random.normalvariate(mean_rri, std_rri)
                
                # Ensure RR interval is physiologically reasonable
                rri = max(0.33, min(1.5, rri))  # Between 40 and 180 BPM
            
            # Add respiratory sinus arrhythmia effect
            respiratory_phase = 2 * np.pi * current_time / 5.0  # ~12s respiratory cycle
            rri += 0.05 * mean_rri * np.sin(respiratory_phase)  # ±5% modulation
            
            # Update time and add peak
            current_time += rri
            if current_time < duration:
                r_peak_times.append(current_time)
        
        return r_peak_times
    
    def _add_arrhythmias(self, signal: np.ndarray, r_peaks: List[int]) -> np.ndarray:
        """Add various arrhythmias to the signal"""
        # Make a copy of the signal
        modified_signal = signal.copy()
        
        # Number of arrhythmias to add
        num_arrhythmias = int(len(r_peaks) * self.arrhythmia_probability)
        
        # Select random locations for arrhythmias
        arrhythmia_locations = random.sample(range(1, len(r_peaks) - 1), num_arrhythmias)
        
        for loc in arrhythmia_locations:
            arrhythmia_type = random.choice(['PVC', 'PAC', 'dropped_beat'])
            
            if arrhythmia_type == 'PVC':
                # Premature Ventricular Contraction
                # Replace a normal beat with a wide, bizarre-shaped beat
                self._add_pvc(modified_signal, r_peaks[loc])
            
            elif arrhythmia_type == 'PAC':
                # Premature Atrial Contraction
                # Similar to normal beats but premature and with abnormal P wave
                self._add_pac(modified_signal, r_peaks[loc], r_peaks[loc-1])
                
            elif arrhythmia_type == 'dropped_beat':
                # Simulate a dropped beat (extended RR interval)
                # No modification needed as the extended RR interval handles this
                pass
        
        return modified_signal
    
    def _add_pvc(self, signal: np.ndarray, peak_idx: int):
        """Add a Premature Ventricular Contraction at the specified index"""
        # PVC template - wider and higher amplitude
        pvc_width = int(self.sampling_rate * 0.15)  # 150ms width
        pvc = np.zeros(pvc_width)
        
        # Create a wide QRS with no P wave
        t = np.linspace(-np.pi, np.pi, pvc_width)
        pvc_qrs = -1.5 * np.exp(-((t)**2) / 0.5)  # Broader, deeper QRS
        pvc += pvc_qrs
        
        # Replace normal beat with PVC
        start_idx = max(0, peak_idx - pvc_width // 2)
        end_idx = min(len(signal), start_idx + pvc_width)
        
        # Zero out the normal beat
        signal[start_idx:end_idx] = 0
        
        # Add PVC
        for i in range(min(pvc_width, end_idx - start_idx)):
            signal[start_idx + i] += pvc[i]
    
    def _add_pac(self, signal: np.ndarray, peak_idx: int, prev_peak_idx: int):
        """Add a Premature Atrial Contraction at the specified index"""
        # PAC is similar to normal beat but with altered P wave
        # Calculate the distance to previous beat
        rr_interval = peak_idx - prev_peak_idx
        
        # Only modify P wave (which comes before the R peak)
        p_wave_start = peak_idx - int(0.2 * self.sampling_rate)  # P wave ~200ms before R
        if p_wave_start > 0:
            # Create an abnormal P wave
            p_wave_len = int(0.1 * self.sampling_rate)  # 100ms P wave
            t = np.linspace(0, np.pi, p_wave_len)
            abnormal_p = 0.3 * np.sin(t)  # Simpler, different P wave
            
            # Replace normal P wave with abnormal one
            for i in range(p_wave_len):
                if p_wave_start + i < len(signal):
                    signal[p_wave_start + i] = abnormal_p[i]
    
    def calculate_hrv_metrics(self) -> Dict:
        """Calculate HRV metrics from generated RR intervals"""
        if self.rr_intervals is None or len(self.rr_intervals) < 2:
            return {}
        
        # Time domain metrics
        mean_nn = np.mean(self.rr_intervals)
        sdnn = np.std(self.rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(self.rr_intervals)**2))
        
        # Calculate pNN50
        nn50 = sum(abs(np.diff(self.rr_intervals)) > 50)
        pnn50 = 100 * nn50 / len(self.rr_intervals[:-1]) if len(self.rr_intervals) > 1 else 0
        
        # Frequency domain metrics would require more complex analysis
        # For simplicity, we only include time domain metrics here
        
        return {
            'mean_nn': float(mean_nn),
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'nn50': int(nn50),
            'pnn50': float(pnn50)
        }
    
    def save_as_csv(self, filename: str):
        """Save generated ECG signal to CSV file"""
        if self.timestamps is None or self.ecg_signal is None:
            raise ValueError("No ECG signal generated. Call generate_ecg() first.")
        
        data = np.column_stack((self.timestamps, self.ecg_signal))
        np.savetxt(filename, data, delimiter=',', header='time,ecg', comments='')
        
        print(f"ECG data saved to {filename}")
    
    def save_with_annotations(self, base_filename: str):
        """Save generated ECG signal with R peak annotations"""
        if self.timestamps is None or self.ecg_signal is None:
            raise ValueError("No ECG signal generated. Call generate_ecg() first.")
        
        # Save ECG signal
        self.save_as_csv(f"{base_filename}.csv")
        
        # Save R peaks
        if self.r_peaks is not None:
            r_peak_times = self.timestamps[self.r_peaks]
            r_peak_data = np.column_stack((r_peak_times, np.ones_like(r_peak_times)))
            np.savetxt(f"{base_filename}_rpeaks.csv", r_peak_data, delimiter=',', 
                      header='time,annotation', comments='')
        
        # Save metadata and HRV metrics
        metadata = {
            'sampling_rate': self.sampling_rate,
            'duration_seconds': self.duration_seconds,
            'heart_rate': self.heart_rate,
            'hrv_level': self.hrv_level,
            'noise_level': self.noise_level,
            'arrhythmia_probability': self.arrhythmia_probability,
            'hrv_metrics': self.calculate_hrv_metrics()
        }
        
        with open(f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"ECG data and annotations saved with base filename: {base_filename}")
    
    def plot_ecg(self, duration: Optional[float] = None, save_path: Optional[str] = None):
        """Plot the generated ECG signal"""
        if self.timestamps is None or self.ecg_signal is None:
            raise ValueError("No ECG signal generated. Call generate_ecg() first.")
        
        plt.figure(figsize=(15, 6))
        
        # If duration specified, plot only that segment
        if duration is not None and duration < self.duration_seconds:
            segment_samples = int(duration * self.sampling_rate)
            plt.plot(self.timestamps[:segment_samples], self.ecg_signal[:segment_samples])
            
            # Mark R peaks if available
            if self.r_peaks is not None:
                # Filter R peaks in the segment
                segment_r_peaks = self.r_peaks[self.r_peaks < segment_samples]
                plt.scatter(self.timestamps[segment_r_peaks], 
                           self.ecg_signal[segment_r_peaks], 
                           color='red', marker='o')
            
            plt.xlim(0, duration)
        else:
            # Plot full signal
            plt.plot(self.timestamps, self.ecg_signal)
            
            # Mark R peaks if available
            if self.r_peaks is not None:
                plt.scatter(self.timestamps[self.r_peaks], 
                           self.ecg_signal[self.r_peaks], 
                           color='red', marker='o')
        
        plt.title(f"Simulated ECG Signal (HR={self.heart_rate}bpm, HRV={self.hrv_level}ms)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ECG plot saved to {save_path}")
        
        plt.show()
    
    def plot_histogram(self, save_path: Optional[str] = None):
        """Plot histogram of RR intervals"""
        if self.rr_intervals is None:
            raise ValueError("No RR intervals available. Call generate_ecg() first.")
        
        plt.figure(figsize=(10, 5))
        plt.hist(self.rr_intervals, bins=30, alpha=0.7, color='skyblue')
        plt.axvline(np.mean(self.rr_intervals), color='red', linestyle='dashed', linewidth=2)
        
        hrv_metrics = self.calculate_hrv_metrics()
        plt.title(f"RR Interval Histogram (SDNN={hrv_metrics.get('sdnn', 0):.1f}ms, RMSSD={hrv_metrics.get('rmssd', 0):.1f}ms)")
        plt.xlabel("RR Interval (ms)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Generate normal sinus rhythm ECG
    ecg_generator = ECGGenerator(
        sampling_rate=250,
        duration_seconds=60,
        heart_rate=70,
        hrv_level=50,
        noise_level=0.03,
        arrhythmia_probability=0.0
    )
    
    # Generate signal
    ecg_generator.generate_ecg()
    
    # Plot 5-second segment
    ecg_generator.plot_ecg(duration=5, save_path="output/normal_ecg.png")
    
    # Plot RR interval histogram
    ecg_generator.plot_histogram(save_path="output/normal_hrv.png")
    
    # Save data with annotations
    ecg_generator.save_with_annotations("output/normal_ecg")
    
    
    # Generate ECG with arrhythmias
    arrhythmia_generator = ECGGenerator(
        sampling_rate=250,
        duration_seconds=60,
        heart_rate=75,
        hrv_level=80,
        noise_level=0.05,
        arrhythmia_probability=0.1
    )
    
    # Generate signal
    arrhythmia_generator.generate_ecg()
    
    # Plot 10-second segment
    arrhythmia_generator.plot_ecg(duration=10, save_path="output/arrhythmia_ecg.png")
    
    # Plot RR interval histogram
    arrhythmia_generator.plot_histogram(save_path="output/arrhythmia_hrv.png")
    
    # Save data with annotations
    arrhythmia_generator.save_with_annotations("output/arrhythmia_ecg")