import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecg_signal_processor")

class SignalProcessor:
    """Processes ECG signals for analysis and transmission"""
    
    def __init__(self):
        # Processing parameters
        self.sampling_rate = 250  # Hz
        self.filter_settings = {
            'highpass': 0.5,  # Hz
            'lowpass': 40.0,  # Hz
            'notch': 50.0     # Hz
        }
        
        # Signal quality metrics
        self.signal_quality = 1.0  # 0.0 (poor) to 1.0 (excellent)
        
        # Filter coefficients
        self.b_hp = None  # Highpass filter coefficients
        self.a_hp = None
        self.b_lp = None  # Lowpass filter coefficients
        self.a_lp = None
        self.b_notch = None  # Notch filter coefficients
        self.a_notch = None
        
        # Filter states
        self.z_hp = None
        self.z_lp = None
        self.z_notch = None
        
        # R-peak detection
        self.r_peak_threshold = 0.6  # Adaptive threshold
        self.last_r_peaks = []  # Store recent R-peak positions
        
        # Detection state
        self.detection_ready = False
        self.buffer_size = 5 * self.sampling_rate  # 5 seconds of data
        self.data_buffer = []  # Buffer for detection algorithms
    
    def initialize(self, sampling_rate: int = 250, filter_settings: Optional[Dict] = None):
        """Initialize the signal processor"""
        logger.info(f"Initializing signal processor with sampling rate {sampling_rate}Hz")
        
        self.sampling_rate = sampling_rate
        
        if filter_settings:
            self.filter_settings.update(filter_settings)
        
        # Initialize filters
        self._initialize_filters()
        
        # Clear buffers
        self.data_buffer = []
        self.last_r_peaks = []
        
        # Reset detection state
        self.detection_ready = False
        self.buffer_size = 5 * self.sampling_rate  # Adjust buffer size based on sampling rate
        
        logger.info("Signal processor initialized successfully")
    
    def _initialize_filters(self):
        """Initialize digital filters"""
        # Highpass filter (remove baseline wander)
        nyquist = 0.5 * self.sampling_rate
        hp_cutoff = self.filter_settings['highpass'] / nyquist
        self.b_hp, self.a_hp = signal.butter(2, hp_cutoff, btype='high')
        self.z_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
        
        # Lowpass filter (remove high-frequency noise)
        lp_cutoff = self.filter_settings['lowpass'] / nyquist
        self.b_lp, self.a_lp = signal.butter(4, lp_cutoff, btype='low')
        self.z_lp = signal.lfilter_zi(self.b_lp, self.a_lp)
        
        # Notch filter (remove power line interference)
        notch_freq = self.filter_settings['notch'] / nyquist
        q_factor = 30.0  # Quality factor of the notch filter
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, q_factor)
        self.z_notch = signal.lfilter_zi(self.b_notch, self.a_notch)
        
        logger.debug("Digital filters initialized")
    
    def update_sampling_rate(self, sampling_rate: int):
        """Update the sampling rate and reinitialize filters"""
        if sampling_rate != self.sampling_rate:
            logger.info(f"Updating sampling rate from {self.sampling_rate}Hz to {sampling_rate}Hz")
            self.sampling_rate = sampling_rate
            
            # Reinitialize filters with new sampling rate
            self._initialize_filters()
            
            # Adjust buffer size
            self.buffer_size = 5 * self.sampling_rate  # 5 seconds of data
            
            # Clear detection state
            self.detection_ready = False
            self.data_buffer = []
            self.last_r_peaks = []
    
    def update_filter_settings(self, filter_settings: Dict):
        """Update filter settings and reinitialize filters"""
        logger.info(f"Updating filter settings: {filter_settings}")
        
        # Update filter settings
        self.filter_settings.update(filter_settings)
        
        # Reinitialize filters with new settings
        self._initialize_filters()
    
    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process ECG data segment"""
        # Extract raw ECG values
        raw_values = [sample['value'] for sample in data]
        
        # Apply filters
        filtered_values, self.z_hp = signal.lfilter(self.b_hp, self.a_hp, raw_values, zi=self.z_hp)
        filtered_values, self.z_notch = signal.lfilter(self.b_notch, self.a_notch, filtered_values, zi=self.z_notch)
        filtered_values, self.z_lp = signal.lfilter(self.b_lp, self.a_lp, filtered_values, zi=self.z_lp)
        
        # Assess signal quality
        signal_quality = self._assess_signal_quality(raw_values, filtered_values)
        self.signal_quality = signal_quality
        
        # Add filtered data to detection buffer
        self.data_buffer.extend(filtered_values)
        
        # Keep buffer at desired size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.detection_ready = True
        
        # Detect R-peaks if enough data is available
        r_peaks = []
        if self.detection_ready:
            r_peaks = self._detect_r_peaks()
        
        # Create processed data with filtered values and additional metrics
        processed_data = []
        for i, (sample, filtered_val) in enumerate(zip(data, filtered_values)):
            processed_sample = sample.copy()
            processed_sample['filtered_value'] = float(filtered_val)
            processed_sample['signal_quality'] = float(signal_quality)
            
            # Mark if this is an R-peak
            processed_sample['is_r_peak'] = False
            for peak_idx in r_peaks:
                if peak_idx == len(self.data_buffer) - len(filtered_values) + i:
                    processed_sample['is_r_peak'] = True
                    break
            
            processed_data.append(processed_sample)
        
        return processed_data
    
    def _assess_signal_quality(self, raw_values: List[float], filtered_values: np.ndarray) -> float:
        """Assess signal quality based on noise level and other factors"""
        # Simple signal quality assessment based on ratio of power in signal vs. noise
        if len(raw_values) < 2:
            return 1.0  # Default to good quality with insufficient data
        
        # Convert to numpy arrays if needed
        raw = np.array(raw_values)
        filtered = np.array(filtered_values)
        
        # Calculate noise as difference between raw and filtered
        noise = raw - filtered
        
        # Calculate signal-to-noise ratio (SNR)
        signal_power = np.var(filtered)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr = signal_power / noise_power
        else:
            snr = 100.0  # Arbitrary high value for zero noise
        
        # Convert SNR to quality metric between 0 and 1
        quality = 1.0 - (1.0 / (1.0 + snr / 3.0))
        
        # Clip to valid range
        quality = min(1.0, max(0.0, quality))
        
        return float(quality)
    
    def _detect_r_peaks(self) -> List[int]:
        """Detect R-peaks in the ECG signal"""
        # Use a simplified R-peak detection algorithm
        # For a real-world implementation, more sophisticated algorithms like Pan-Tompkins would be used
        
        # Get last second of data for analysis
        analysis_window = self.data_buffer[-self.sampling_rate:]
        
        # Apply adaptive threshold for peak detection
        signal_max = np.max(np.abs(analysis_window))
        threshold = signal_max * self.r_peak_threshold
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(analysis_window, height=threshold, distance=int(0.2 * self.sampling_rate))
        
        # Convert to indices in the full buffer
        peak_indices = [len(self.data_buffer) - len(analysis_window) + peak for peak in peaks]
        
        # Store detected peaks for future reference
        self.last_r_peaks.extend(peak_indices)
        self.last_r_peaks = self.last_r_peaks[-10:]  # Keep last 10 peaks
        
        # Adapt threshold based on detection results
        if len(peaks) > 0:
            peak_heights = [analysis_window[p] for p in peaks]
            avg_peak_height = np.mean(peak_heights)
            # Gradually adjust threshold
            self.r_peak_threshold = 0.9 * self.r_peak_threshold + 0.1 * (0.6 * avg_peak_height / signal_max)
        
        return peak_indices
    
    def calculate_heart_rate(self) -> Dict:
        """Calculate heart rate from detected R-peaks"""
        if len(self.last_r_peaks) < 2:
            return {
                'heart_rate': 0,
                'confidence': 0,
                'rr_intervals': []
            }
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff([peak / self.sampling_rate for peak in self.last_r_peaks])
        
        # Filter out physiologically impossible intervals
        valid_intervals = [rr for rr in rr_intervals if 0.3 <= rr <= 2.0]  # 30-200 bpm range
        
        if len(valid_intervals) < 1:
            return {
                'heart_rate': 0,
                'confidence': 0,
                'rr_intervals': []
            }
        
        # Calculate heart rate from average RR interval
        avg_rr = np.mean(valid_intervals)
        heart_rate = 60.0 / avg_rr
        
        # Calculate confidence based on RR interval variability and signal quality
        rr_std = np.std(valid_intervals) if len(valid_intervals) > 1 else 0
        rr_confidence = 1.0 - min(1.0, rr_std / avg_rr)  # Lower variability = higher confidence
        
        # Overall confidence combines signal quality and RR interval stability
        confidence = 0.7 * self.signal_quality + 0.3 * rr_confidence
        
        return {
            'heart_rate': float(heart_rate),
            'confidence': float(confidence),
            'rr_intervals': [float(rr) for rr in valid_intervals]
        }