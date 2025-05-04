import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waveform_generator")

class StimulationMode(Enum):
    CONTINUOUS = "continuous"
    BURST = "burst"
    INTERMITTENT = "intermittent"

class WaveformGenerator:
    """Generates various waveforms for TENS stimulation"""
    
    def __init__(self):
        # Default parameters
        self.frequency = 80.0  # Hz
        self.pulse_width = 200.0  # µs
        self.intensity = 0.0  # mA
        self.mode = StimulationMode.CONTINUOUS
        
        # Waveform shape parameters
        self.wave_shape = "biphasic"  # "monophasic", "biphasic", "asymmetric"
        
        # Mode-specific parameters
        self.burst_frequency = 2.0  # Hz (for burst mode)
        self.burst_duty_cycle = 0.5  # 50% (for burst mode)
        self.intermittent_on_time = 10.0  # seconds (for intermittent mode)
        self.intermittent_off_time = 5.0  # seconds (for intermittent mode)
        
        # Internal state
        self.last_update_time = time.time()
        self.total_time = 0.0
        self.sample_rate = 10000.0  # Hz (internal sample rate for waveform generation)
        
        logger.info("Waveform generator initialized")
    
    def initialize(self, frequency: float, pulse_width: float, intensity: float, mode: StimulationMode):
        """Initialize the waveform generator with specific parameters"""
        self.frequency = frequency
        self.pulse_width = pulse_width
        self.intensity = intensity
        self.mode = mode
        
        self.last_update_time = time.time()
        self.total_time = 0.0
        
        logger.info(f"Waveform generator initialized with: frequency={frequency}Hz, "
                   f"pulse_width={pulse_width}µs, intensity={intensity}mA, mode={mode.value}")
    
    def update_parameters(self, frequency: Optional[float] = None, 
                         pulse_width: Optional[float] = None,
                         intensity: Optional[float] = None,
                         mode: Optional[StimulationMode] = None):
        """Update waveform parameters"""
        if frequency is not None:
            self.frequency = frequency
        
        if pulse_width is not None:
            self.pulse_width = pulse_width
        
        if intensity is not None:
            self.intensity = intensity
        
        if mode is not None:
            self.mode = mode
        
        logger.debug(f"Updated waveform parameters: frequency={self.frequency}Hz, "
                    f"pulse_width={self.pulse_width}µs, intensity={self.intensity}mA, mode={self.mode.value}")
    
    def set_mode_parameters(self, burst_frequency: Optional[float] = None,
                          burst_duty_cycle: Optional[float] = None,
                          intermittent_on_time: Optional[float] = None,
                          intermittent_off_time: Optional[float] = None):
        """Set parameters specific to stimulation modes"""
        if burst_frequency is not None:
            self.burst_frequency = burst_frequency
        
        if burst_duty_cycle is not None:
            self.burst_duty_cycle = max(0.1, min(0.9, burst_duty_cycle))  # Ensure between 10-90%
        
        if intermittent_on_time is not None:
            self.intermittent_on_time = intermittent_on_time
        
        if intermittent_off_time is not None:
            self.intermittent_off_time = intermittent_off_time
    
    def set_wave_shape(self, shape: str):
        """Set the shape of the waveform"""
        valid_shapes = ["monophasic", "biphasic", "asymmetric"]
        
        if shape.lower() in valid_shapes:
            self.wave_shape = shape.lower()
            logger.info(f"Waveform shape set to {shape}")
        else:
            logger.warning(f"Invalid wave shape: {shape}. Using default 'biphasic'.")
            self.wave_shape = "biphasic"
    
    def generate_pulse(self) -> float:
        """Generate a single TENS pulse value based on current parameters"""
        # Get current time and update total time
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        self.total_time += elapsed
        
        # Calculate pulse based on mode
        if self.mode == StimulationMode.CONTINUOUS:
            return self._generate_continuous_pulse()
        elif self.mode == StimulationMode.BURST:
            return self._generate_burst_pulse()
        elif self.mode == StimulationMode.INTERMITTENT:
            return self._generate_intermittent_pulse()
        else:
            return 0.0
    
    def generate_waveform(self, duration_ms: float = 10.0) -> np.ndarray:
        """Generate a waveform for a specified duration in milliseconds"""
        # Calculate number of samples
        num_samples = int((duration_ms / 1000.0) * self.sample_rate)
        
        # Generate time array
        t = np.linspace(0, duration_ms / 1000.0, num_samples) + self.total_time
        
        # Update total time
        self.total_time += duration_ms / 1000.0
        self.last_update_time = time.time()
        
        # Create waveform array
        waveform = np.zeros(num_samples)
        
        # Generate waveform based on mode
        if self.mode == StimulationMode.CONTINUOUS:
            waveform = self._generate_continuous_waveform(t)
        elif self.mode == StimulationMode.BURST:
            waveform = self._generate_burst_waveform(t)
        elif self.mode == StimulationMode.INTERMITTENT:
            waveform = self._generate_intermittent_waveform(t)
        
        return waveform
    
    def _generate_continuous_pulse(self) -> float:
        """Generate a single value for continuous mode"""
        # Calculate phase in the pulse cycle
        cycle_time = 1.0 / self.frequency
        phase = (self.total_time % cycle_time) / cycle_time
        
        # Calculate pulse value based on phase
        pulse_duration_ratio = self.pulse_width / 1000000.0 / cycle_time  # pulse_width is in µs
        
        if phase < pulse_duration_ratio:
            # During pulse
            if self.wave_shape == "monophasic":
                return self.intensity
            elif self.wave_shape == "biphasic":
                if phase < pulse_duration_ratio / 2.0:
                    return self.intensity
                else:
                    return -self.intensity
            elif self.wave_shape == "asymmetric":
                if phase < pulse_duration_ratio * 0.75:
                    return self.intensity
                else:
                    return -self.intensity * 0.5
        
        # No pulse
        return 0.0
    
    def _generate_burst_pulse(self) -> float:
        """Generate a single value for burst mode"""
        # Calculate if we're in a burst
        burst_cycle_time = 1.0 / self.burst_frequency
        burst_phase = (self.total_time % burst_cycle_time) / burst_cycle_time
        
        if burst_phase < self.burst_duty_cycle:
            # During burst, generate continuous pulses
            return self._generate_continuous_pulse()
        else:
            # Between bursts
            return 0.0
    
    def _generate_intermittent_pulse(self) -> float:
        """Generate a single value for intermittent mode"""
        # Calculate if we're in an ON period
        cycle_time = self.intermittent_on_time + self.intermittent_off_time
        phase = (self.total_time % cycle_time) / cycle_time
        
        if phase < self.intermittent_on_time / cycle_time:
            # During ON period, generate continuous pulses
            return self._generate_continuous_pulse()
        else:
            # During OFF period
            return 0.0
    
    def _generate_continuous_waveform(self, t: np.ndarray) -> np.ndarray:
        """Generate a continuous mode waveform for the given time array"""
        # Calculate the period of one pulse cycle
        cycle_time = 1.0 / self.frequency
        
        # Calculate the pulse width in seconds
        pulse_width_sec = self.pulse_width / 1000000.0  # convert from µs to s
        
        # Generate waveform based on shape
        waveform = np.zeros_like(t)
        
        if self.wave_shape == "monophasic":
            # Generate monophasic pulses
            for i, time_val in enumerate(t):
                phase = (time_val % cycle_time) / cycle_time
                if phase < pulse_width_sec / cycle_time:
                    waveform[i] = self.intensity
        
        elif self.wave_shape == "biphasic":
            # Generate biphasic pulses
            for i, time_val in enumerate(t):
                phase = (time_val % cycle_time) / cycle_time
                pulse_duration_ratio = pulse_width_sec / cycle_time
                
                if phase < pulse_duration_ratio:
                    if phase < pulse_duration_ratio / 2.0:
                        waveform[i] = self.intensity
                    else:
                        waveform[i] = -self.intensity
        
        elif self.wave_shape == "asymmetric":
            # Generate asymmetric biphasic pulses
            for i, time_val in enumerate(t):
                phase = (time_val % cycle_time) / cycle_time
                pulse_duration_ratio = pulse_width_sec / cycle_time
                
                if phase < pulse_duration_ratio:
                    if phase < pulse_duration_ratio * 0.75:
                        waveform[i] = self.intensity
                    else:
                        waveform[i] = -self.intensity * 0.5
        
        return waveform
    
    def _generate_burst_waveform(self, t: np.ndarray) -> np.ndarray:
        """Generate a burst mode waveform for the given time array"""
        # First create a continuous waveform
        continuous_waveform = self._generate_continuous_waveform(t)
        
        # Then apply burst modulation
        burst_waveform = np.zeros_like(continuous_waveform)
        burst_cycle_time = 1.0 / self.burst_frequency
        
        for i, time_val in enumerate(t):
            burst_phase = (time_val % burst_cycle_time) / burst_cycle_time
            if burst_phase < self.burst_duty_cycle:
                burst_waveform[i] = continuous_waveform[i]
        
        return burst_waveform
    
    def _generate_intermittent_waveform(self, t: np.ndarray) -> np.ndarray:
        """Generate an intermittent mode waveform for the given time array"""
        # First create a continuous waveform
        continuous_waveform = self._generate_continuous_waveform(t)
        
        # Then apply intermittent modulation
        intermittent_waveform = np.zeros_like(continuous_waveform)
        cycle_time = self.intermittent_on_time + self.intermittent_off_time
        
        for i, time_val in enumerate(t):
            phase = (time_val % cycle_time) / cycle_time
            if phase < self.intermittent_on_time / cycle_time:
                intermittent_waveform[i] = continuous_waveform[i]
        
        return intermittent_waveform