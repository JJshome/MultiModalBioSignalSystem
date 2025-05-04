import numpy as np
import time
import random
from typing import Dict, List, Optional, Any
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppg_sensor")

class PPGSensor:
    """Interface for PPG sensor hardware"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        
        # Sensor configuration
        self.sampling_rate = 100  # Hz
        self.led_intensity = 20  # mA
        self.led_wavelength = 525  # nm (green light)
        
        # Sensor status
        self.initialized = False
        self.status = 'IDLE'
        
        # Simulation parameters
        if simulation_mode:
            self.sim_heart_rate = 65  # bpm
            self.sim_hrv = 50  # ms (SDNN)
            self.sim_respiratory_rate = 15  # breaths per minute
            self.sim_motion_artifact = 0.1  # Level of motion artifact (0-1)
            self.sim_signal_quality = 0.9  # Signal quality (0-1)
            
            self.sim_time = 0.0
            self.sim_last_update = time.time()
            
            # Create PPG waveform template
            self.ppg_template = self._create_ppg_template()
            
            # Slow variation parameters for simulation
            self.hr_drift_period = 300  # seconds for HR to complete one cycle
            self.respiratory_influence = 0.1  # Influence of respiration on HR
    
    def initialize(self, sampling_rate: int = 100) -> Dict:
        """Initialize the PPG sensor"""
        logger.info(f"Initializing PPG sensor with sampling rate {sampling_rate}Hz")
        
        self.sampling_rate = sampling_rate
        
        if self.simulation_mode:
            # Reset simulation parameters
            self.sim_time = 0.0
            self.sim_last_update = time.time()
            
            # Randomize starting heart rate between 60-80 bpm
            self.sim_heart_rate = random.uniform(60, 80)
            
            self.initialized = True
            self.status = 'OK'
            
            logger.info("PPG sensor initialized (simulation mode)")
            return {'success': True, 'message': 'Sensor initialized in simulation mode'}
        else:
            # Real hardware initialization would go here
            try:
                # This would be replaced with actual hardware initialization code
                logger.info("Initializing real PPG sensor hardware")
                
                # Placeholder for real initialization
                self.initialized = True
                self.status = 'OK'
                
                logger.info("PPG sensor initialized successfully")
                return {'success': True, 'message': 'Sensor initialized successfully'}
            except Exception as e:
                logger.error(f"Failed to initialize PPG sensor: {e}")
                self.status = 'ERROR'
                self.initialized = False
                return {'success': False, 'message': f"Sensor initialization failed: {str(e)}"}
    
    def shutdown(self):
        """Shut down the PPG sensor"""
        logger.info("Shutting down PPG sensor")
        
        if not self.simulation_mode:
            # Real hardware shutdown would go here
            pass
        
        self.initialized = False
        self.status = 'IDLE'
        
        logger.info("PPG sensor shut down successfully")
    
    def set_sampling_rate(self, rate: int):
        """Set the sampling rate"""
        if rate != self.sampling_rate:
            logger.info(f"Changing sampling rate from {self.sampling_rate}Hz to {rate}Hz")
            self.sampling_rate = rate
            
            if not self.simulation_mode:
                # Real hardware sampling rate adjustment would go here
                pass
    
    def set_led_intensity(self, intensity: int):
        """Set the LED intensity"""
        if intensity != self.led_intensity:
            logger.info(f"Changing LED intensity from {self.led_intensity}mA to {intensity}mA")
            self.led_intensity = intensity
            
            if not self.simulation_mode:
                # Real hardware LED intensity adjustment would go here
                pass
    
    def read_sample(self) -> Dict:
        """Read a single PPG sample"""
        if not self.initialized:
            logger.warning("Attempting to read from uninitialized sensor")
            return {'timestamp': time.time(), 'value': 0, 'status': 'ERROR'}
        
        if self.simulation_mode:
            # Generate simulated PPG data
            sample = self._generate_simulated_sample()
            return sample
        else:
            # Real hardware sample reading would go here
            try:
                # This would be replaced with actual hardware reading code
                timestamp = time.time()
                value = 0.0  # Placeholder
                
                return {'timestamp': timestamp, 'value': value, 'status': 'OK'}
            except Exception as e:
                logger.error(f"Error reading PPG sample: {e}")
                return {'timestamp': time.time(), 'value': 0, 'status': 'ERROR'}
    
    def test_sensor(self) -> Dict:
        """Test the PPG sensor"""
        logger.info("Testing PPG sensor")
        
        if self.simulation_mode:
            # In simulation mode, always return success
            signal_quality = random.uniform(0.8, 1.0)
            ambient_light = random.uniform(20, 100)  # Arbitrary units
            
            return {
                'success': True,
                'status': 'OK',
                'signal_quality': signal_quality,
                'ambient_light': ambient_light,
                'led_test': True,
                'message': 'Sensor test successful (simulation)'
            }
        else:
            # Real hardware sensor test would go here
            try:
                # This would be replaced with actual hardware test code
                
                # Placeholder values
                signal_quality = 0.9
                ambient_light = 50
                
                return {
                    'success': True,
                    'status': 'OK',
                    'signal_quality': signal_quality,
                    'ambient_light': ambient_light,
                    'led_test': True,
                    'message': 'Sensor test successful'
                }
            except Exception as e:
                logger.error(f"Error testing PPG sensor: {e}")
                return {
                    'success': False,
                    'status': 'ERROR',
                    'signal_quality': 0,
                    'ambient_light': 0,
                    'led_test': False,
                    'message': f"Sensor test failed: {str(e)}"
                }
    
    def _create_ppg_template(self) -> np.ndarray:
        """Create a template for simulated PPG waveform"""
        # Create a realistic PPG template
        t = np.linspace(0, 2*np.pi, 100)
        
        # Main pulse wave (systolic peak)
        systolic_peak = np.exp(-((t - 1.0)**2) / 0.1)
        
        # Dicrotic notch and diastolic peak
        dicrotic_notch = -0.2 * np.exp(-((t - 2.0)**2) / 0.02)
        diastolic_peak = 0.3 * np.exp(-((t - 2.3)**2) / 0.1)
        
        # Combine components
        ppg_wave = systolic_peak + dicrotic_notch + diastolic_peak
        
        # Add baseline
        baseline = 0.5 * np.ones_like(t)
        ppg_template = baseline + ppg_wave
        
        # Normalize to 0-1 range
        ppg_template = (ppg_template - np.min(ppg_template)) / (np.max(ppg_template) - np.min(ppg_template))
        
        return ppg_template
    
    def _generate_simulated_sample(self) -> Dict:
        """Generate a simulated PPG sample"""
        current_time = time.time()
        dt = current_time - self.sim_last_update
        self.sim_last_update = current_time
        
        # Update simulation time
        self.sim_time += dt
        
        # Add natural heart rate variability
        # 1. Slow drift in baseline heart rate
        hr_drift = 5 * math.sin(2 * math.pi * self.sim_time / self.hr_drift_period)  # ±5 bpm drift
        
        # 2. Respiratory influence (respiratory sinus arrhythmia)
        respiratory_phase = 2 * math.pi * self.sim_time * self.sim_respiratory_rate / 60
        respiratory_effect = self.respiratory_influence * math.sin(respiratory_phase) * self.sim_heart_rate
        
        # 3. Random beat-to-beat variability
        random_variation = random.gauss(0, self.sim_hrv / 1000 * self.sim_heart_rate)
        
        # Combine effects for current heart rate
        current_hr = self.sim_heart_rate + hr_drift + respiratory_effect + random_variation
        current_hr = max(40, min(180, current_hr))  # Limit to physiological range
        
        # Calculate period and phase
        period = 60.0 / current_hr  # seconds per beat
        phase = (self.sim_time % period) / period
        
        # Get template index
        template_index = int(phase * len(self.ppg_template))
        
        # Get PPG value from template
        ppg_value = self.ppg_template[template_index]
        
        # Add noise and artifacts
        # 1. Basic sensor noise
        noise = 0.01 * random.gauss(0, 1)
        
        # 2. Motion artifacts (occasional larger disturbances)
        if random.random() < 0.05:  # 5% chance of motion artifact
            artifact = self.sim_motion_artifact * random.uniform(-1, 1)
        else:
            artifact = 0
        
        # Combine signal components
        ppg_value = ppg_value + noise + artifact
        
        # Ensure value is within reasonable range
        ppg_value = max(0, min(1, ppg_value))
        
        # Occasionally update simulation parameters for realism
        if random.random() < 0.001:  # 0.1% chance per sample
            # Gradual changes to heart rate
            hr_change = random.uniform(-2, 2)  # ±2 bpm
            self.sim_heart_rate = max(50, min(100, self.sim_heart_rate + hr_change))
            
            # Occasional changes to signal quality
            quality_change = random.uniform(-0.05, 0.05)
            self.sim_signal_quality = max(0.5, min(1.0, self.sim_signal_quality + quality_change))
            
            # Occasional changes to motion artifact level
            artifact_change = random.uniform(-0.05, 0.05)
            self.sim_motion_artifact = max(0.05, min(0.3, self.sim_motion_artifact + artifact_change))
        
        # Create sample dict
        sample = {
            'timestamp': current_time,
            'value': float(ppg_value),
            'quality': float(self.sim_signal_quality - abs(artifact)),  # Reduce quality during artifacts
            'status': 'OK'
        }
        
        return sample