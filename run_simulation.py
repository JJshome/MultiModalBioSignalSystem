#!/usr/bin/env python3

import argparse
import time
import logging
import threading
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import random

# Import system components
from central_system.bluetooth_manager import BluetoothManager
from central_system.data_processor import DataProcessor
from central_system.ai_controller import AIController
from central_system.config import SystemConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation")

class SimulationManager:
    """Manages simulation of the multi-modal biosignal system"""
    
    def __init__(self, config_path: str = None, output_dir: str = "simulation_output"):
        # Load configuration
        self.config = SystemConfig()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Ensure simulation mode is on
        self.config.simulation_mode = True
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.bluetooth_manager = BluetoothManager(self.config.bluetooth_config)
        self.data_processor = DataProcessor(
            sampling_rate_ecg=self.config.data_processing_config.ecg_sampling_rate,
            sampling_rate_hrv=self.config.data_processing_config.hrv_sampling_rate,
            window_size_sec=self.config.data_processing_config.window_size_sec,
            overlap_sec=self.config.data_processing_config.overlap_sec
        )
        self.ai_controller = AIController(
            model_path=self.config.ai_control_config.model_directory,
            use_reinforcement_learning=self.config.ai_control_config.use_reinforcement_learning
        )
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Simulation parameters
        self.simulation_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed
        self.ecg_device_connected = False
        self.hrv_device_connected = False
        self.tens_connected = False
        
        # Data recording
        self.recording = False
        self.recorded_data = {
            'ecg': [],
            'hrv': [],
            'tens_parameters': [],
            'features': [],
            'timestamps': []
        }
    
    def _load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Apply configuration
            # This would need to be implemented based on your config structure
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def start_simulation(self, duration_seconds: int = 300):
        """Start the simulation"""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        logger.info(f"Starting simulation for {duration_seconds} seconds")
        
        # Reset state
        self.stop_event.clear()
        self.is_running = True
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(duration_seconds,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start recording data
        self.start_recording()
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.is_running:
            logger.warning("Simulation is not running")
            return
        
        logger.info("Stopping simulation")
        
        # Signal simulation to stop
        self.stop_event.set()
        
        # Wait for simulation thread to end
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
        
        # Update state
        self.is_running = False
        
        # Stop recording and save data
        self.stop_recording()
    
    def start_recording(self):
        """Start recording simulation data"""
        logger.info("Starting data recording")
        self.recording = True
        self.recorded_data = {
            'ecg': [],
            'hrv': [],
            'tens_parameters': [],
            'features': [],
            'timestamps': []
        }
    
    def stop_recording(self):
        """Stop recording and save data"""
        if not self.recording:
            return
        
        logger.info("Stopping data recording")
        self.recording = False
        
        # Save recorded data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"simulation_data_{timestamp}.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.recorded_data, f, indent=2)
            logger.info(f"Saved simulation data to {output_file}")
            
            # Generate plots
            self._generate_plots(timestamp)
        except Exception as e:
            logger.error(f"Error saving simulation data: {e}")
    
    def _generate_plots(self, timestamp: str):
        """Generate plots from simulation data"""
        try:
            # Create figure for ECG and HRV data
            plt.figure(figsize=(12, 8))
            
            # Plot ECG data
            if self.recorded_data['ecg']:
                plt.subplot(3, 1, 1)
                ecg_values = [sample.get('filtered_value', sample.get('value', 0)) 
                             for sample in self.recorded_data['ecg']]
                plt.plot(ecg_values[-500:])  # Plot last 500 samples
                plt.title('ECG Signal (Last 500 Samples)')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
            
            # Plot heart rate
            if self.recorded_data['features']:
                plt.subplot(3, 1, 2)
                heart_rates = [feature.get('heart_rate', 0) for feature in self.recorded_data['features']]
                plt.plot(heart_rates)
                plt.title('Heart Rate Over Time')
                plt.xlabel('Time Point')
                plt.ylabel('Heart Rate (BPM)')
            
            # Plot HRV (RMSSD)
            if self.recorded_data['features']:
                plt.subplot(3, 1, 3)
                rmssd_values = [feature.get('rmssd', 0) for feature in self.recorded_data['features']]
                plt.plot(rmssd_values)
                plt.title('HRV (RMSSD) Over Time')
                plt.xlabel('Time Point')
                plt.ylabel('RMSSD (ms)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"simulation_plots_{timestamp}.png"))
            plt.close()
            
            logger.info(f"Generated plots to simulation_plots_{timestamp}.png")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _simulation_loop(self, duration_seconds: int):
        """Main simulation loop"""
        logger.info("Simulation loop started")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Simulate device connections
        time.sleep(1.0)  # Simulate connection time
        self.ecg_device_connected = True
        self.hrv_device_connected = True
        self.tens_connected = True
        logger.info("Simulated devices connected")
        
        # Main simulation loop
        while not self.stop_event.is_set() and time.time() < end_time:
            loop_start = time.time()
            
            # 1. Simulate ECG data reception
            if self.ecg_device_connected:
                ecg_data = self._generate_ecg_data()
                self.data_processor.add_ecg_data(np.array(ecg_data), datetime.now())
                
                if self.recording:
                    self.recorded_data['ecg'].extend(ecg_data)
            
            # 2. Simulate HRV data reception
            if self.hrv_device_connected:
                hrv_data = self._generate_hrv_data()
                self.data_processor.add_ppg_data(np.array(hrv_data), datetime.now())
                
                if self.recording:
                    self.recorded_data['hrv'].extend(hrv_data)
            
            # 3. Process data
            ecg_features, hrv_features = self.data_processor.process_windows()
            
            # 4. Detect anomalies
            anomaly_detected, anomaly_score, anomaly_details = self.data_processor.detect_anomalies()
            
            # 5. Get combined features for AI controller
            combined_features = self.data_processor.get_combined_features()
            
            if self.recording and combined_features:
                self.recorded_data['features'].append(combined_features)
                self.recorded_data['timestamps'].append(time.time())
            
            # 6. Get TENS parameters from AI controller
            if combined_features:
                tens_params = self.ai_controller.process_features(combined_features)
                
                if self.recording:
                    self.recorded_data['tens_parameters'].append(tens_params)
                
                # 7. Update AI controller with response (in a real system, this would come from actual measurements)
                # Here we simulate a response after a delay
                if random.random() < 0.2:  # 20% chance to update with response
                    # Simulate improved HRV after stimulation
                    improved_features = combined_features.copy()
                    if tens_params.get('intensity', 0) > 10:
                        # Simulate improvement proportional to stimulation intensity
                        intensity = tens_params.get('intensity', 0)
                        improvement_factor = min(0.3, intensity / 100)  # Max 30% improvement
                        
                        # Improve relevant metrics
                        improved_features['rmssd'] = combined_features.get('rmssd', 30) * (1 + improvement_factor)
                        improved_features['sdnn'] = combined_features.get('sdnn', 50) * (1 + improvement_factor)
                        
                        # Simulate moving closer to optimal LF/HF ratio
                        current_lfhf = combined_features.get('lf_hf_ratio', 1.5)
                        optimal_lfhf = 1.5  # Balanced autonomic tone
                        improved_features['lf_hf_ratio'] = current_lfhf + (optimal_lfhf - current_lfhf) * improvement_factor
                        
                        # Update AI controller with response
                        self.ai_controller.update_with_response(improved_features)
            
            # Calculate sleep time to maintain simulation speed
            loop_time = time.time() - loop_start
            target_loop_time = 0.1 / self.simulation_speed  # 10 Hz update rate adjusted for speed
            sleep_time = max(0, target_loop_time - loop_time)
            time.sleep(sleep_time)
        
        logger.info("Simulation loop ended")
    
    def _generate_ecg_data(self, num_samples: int = 50) -> List[Dict]:
        """Generate simulated ECG data"""
        # Use a simple sine wave with noise as a very basic ECG simulation
        # In a real implementation, this would be more sophisticated
        
        data = []
        base_hr = 60 + 20 * np.sin(time.time() / 60)  # Heart rate varies between 40-80 BPM
        
        for i in range(num_samples):
            # Simple ECG-like waveform
            t = time.time() + i / self.config.data_processing_config.ecg_sampling_rate
            phase = (t % (60 / base_hr)) / (60 / base_hr) * 2 * np.pi
            
            # Basic ECG shape with QRS complex
            if phase < 0.1 * 2 * np.pi:
                # P wave
                value = 0.2 * np.sin(phase / 0.1)
            elif phase < 0.2 * 2 * np.pi:
                # PR segment
                value = 0
            elif phase < 0.3 * 2 * np.pi:
                # QRS complex
                value = -0.5 * np.sin(phase / 0.05) + 1.5 * np.sin((phase - 0.25) / 0.05)
            elif phase < 0.7 * 2 * np.pi:
                # ST segment
                value = 0
            elif phase < 0.9 * 2 * np.pi:
                # T wave
                value = 0.3 * np.sin((phase - 0.7) / 0.2)
            else:
                value = 0
            
            # Add noise
            value += 0.05 * np.random.normal()
            
            # Create sample
            sample = {
                'timestamp': time.time() + i / self.config.data_processing_config.ecg_sampling_rate,
                'value': float(value),
                'filtered_value': float(value)  # In real data, this would be different
            }
            
            data.append(sample)
        
        return data
    
    def _generate_hrv_data(self, num_samples: int = 20) -> List[Dict]:
        """Generate simulated HRV (PPG) data"""
        data = []
        base_hr = 60 + 20 * np.sin(time.time() / 60)  # Heart rate varies between 40-80 BPM
        
        for i in range(num_samples):
            # Simple PPG-like waveform
            t = time.time() + i / self.config.data_processing_config.hrv_sampling_rate
            phase = (t % (60 / base_hr)) / (60 / base_hr) * 2 * np.pi
            
            # Basic PPG pulse shape
            value = 0.5 + 0.4 * np.exp(-((phase - 0.5) ** 2) / 0.1) + 0.1 * np.exp(-((phase - 0.7) ** 2) / 0.05)
            
            # Add noise
            value += 0.03 * np.random.normal()
            
            # Create sample
            sample = {
                'timestamp': t,
                'value': float(value),
                'quality': 0.9 + 0.1 * np.random.random()  # Random quality between 0.9-1.0
            }
            
            data.append(sample)
        
        return data


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run MultiModal BioSignal System Simulation')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    
    parser.add_argument('--output', type=str, default='simulation_output',
                        help='Directory to save output data and plots')
    
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration of simulation in seconds')
    
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Simulation speed multiplier (1.0 = real-time)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create simulation manager
    sim_manager = SimulationManager(
        config_path=args.config,
        output_dir=args.output
    )
    
    # Set simulation speed
    sim_manager.simulation_speed = args.speed
    
    try:
        # Run simulation
        sim_manager.start_simulation(duration_seconds=args.duration)
        
        # Wait for simulation to finish
        while sim_manager.is_running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Stopping simulation...")
                sim_manager.stop_simulation()
                break
    
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
    finally:
        # Ensure simulation is stopped
        if sim_manager.is_running:
            sim_manager.stop_simulation()
    
    logger.info("Simulation complete")


if __name__ == "__main__":
    main()