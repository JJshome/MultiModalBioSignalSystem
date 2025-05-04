"""Configuration settings for the multi-modal bio-signal system"""

class BluetoothConfig:
    """Bluetooth communication configuration"""
    # Connection settings
    connection_timeout = 10.0  # seconds
    connection_monitor_interval = 5.0  # seconds
    
    # Device characteristic UUIDs
    ecg_characteristic_uuid = "00002a37-0000-1000-8000-00805f9b34fb"  # Heart Rate Measurement
    hrv_characteristic_uuid = "00002a37-0000-1000-8000-00805f9b34fb"  # Same for HRV as it uses HR data
    tens_characteristic_uuid = "00002a39-0000-1000-8000-00805f9b34fb"  # Custom UUID for TENS control
    
    # Data format settings
    ecg_data_format = "float32"  # Data type for ECG values
    hrv_data_format = "float32"  # Data type for HRV values
    
    # Maximum packet size
    max_packet_size = 512  # bytes
    
    # Synchronization settings
    max_time_deviation = 0.5  # maximum allowed time deviation in seconds


class DataProcessingConfig:
    """Data processing and analysis configuration"""
    # Sampling rates
    ecg_sampling_rate = 250  # Hz
    hrv_sampling_rate = 100  # Hz
    
    # Window and overlap for analysis
    window_size_sec = 10.0  # seconds
    overlap_sec = 5.0  # seconds
    
    # Filter settings
    ecg_lowpass_freq = 40.0  # Hz
    ecg_highpass_freq = 0.5  # Hz
    ppg_lowpass_freq = 8.0  # Hz
    ppg_highpass_freq = 0.5  # Hz
    notch_freq = 50.0  # Hz (for power line interference)
    
    # HRV analysis settings
    min_peaks_for_hrv = 5  # Minimum number of peaks needed for HRV analysis


class TENSConfig:
    """TENS stimulation configuration"""
    # Output limits
    max_frequency = 120  # Hz
    min_frequency = 20  # Hz
    max_pulse_width = 300  # µs
    min_pulse_width = 50  # µs
    max_intensity = 50  # mA
    min_intensity = 0  # mA
    
    # Stimulation modes
    available_modes = ["continuous", "burst", "intermittent"]
    
    # Safety settings
    max_session_duration = 60  # minutes
    min_rest_between_sessions = 60  # minutes
    emergency_stop_timeout = 0.5  # seconds - maximum delay for emergency stop


class AIControlConfig:
    """AI control system configuration"""
    # Learning settings
    use_reinforcement_learning = True
    gamma = 0.99  # Discount factor
    tau = 0.001  # Target network update rate
    batch_size = 64
    buffer_size = 10000
    exploration_noise = 0.1
    
    # Model update frequency
    update_frequency = 5  # Update models every 5 samples
    
    # Feature importance
    feature_weights = {
        'heart_rate': 1.0,
        'rmssd': 1.0,
        'sdnn': 0.8,
        'lf_hf_ratio': 1.0,
        'hf_power': 0.7,
        'lf_power': 0.7,
        'sd1': 0.9,
        'sd2': 0.8,
        'qt_interval': 0.6,
        'pq_interval': 0.5,
        'ecg_quality': 0.5,
        'anomaly_score': 1.0
    }
    
    # Model paths
    model_directory = "models/"


class SystemConfig:
    """Overall system configuration"""
    # Device names
    ecg_device_name = "ECG-Monitor"
    hrv_device_name = "HRV-Monitor"
    tens_device_name = "TENS-Stimulator"
    
    # Data storage
    data_directory = "data/"
    log_directory = "logs/"
    
    # User interface
    ui_refresh_rate = 0.5  # seconds
    
    # Component configurations
    bluetooth_config = BluetoothConfig()
    data_processing_config = DataProcessingConfig()
    tens_config = TENSConfig()
    ai_control_config = AIControlConfig()
    
    # System operation mode
    simulation_mode = True  # Set to False for real device operation
    debug_mode = True  # Set to False for production