#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Analysis Integration Module

This module integrates the transformer-based anomaly detection system with the 
central control system of the multi-modal bio-signal platform. It provides interfaces
for real-time anomaly detection, pattern recognition, and advanced analysis of
bio-signals collected from various wireless devices.

Based on the patent "Multi-Modal Bio-Stimulation, Diagnosis and Treatment System
Using Multiple Wireless Devices and Method for the Same"
"""

import os
import time
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

# Import the transformer model
from ai_models.transformer_anomaly_detector import BioSignalAnomalyDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transformer_analysis")

class TransformerAnalysisModule:
    """
    Central integration module for transformer-based bio-signal analysis
    
    This class provides methods to:
    1. Initialize and manage transformer models for different signal types
    2. Process incoming bio-signals in real-time
    3. Detect anomalies and patterns in the signals
    4. Provide feedback for stimulation parameter adjustment
    5. Generate comprehensive analysis reports
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        signal_types: List[str] = ["ecg", "emg", "ppg", "eeg", "gsr", "acc"],
        window_sizes: Dict[str, int] = {
            "ecg": 500,   # 2.5 seconds at 200Hz
            "emg": 200,   # 1 second at 200Hz
            "ppg": 100,   # 2 seconds at 50Hz
            "eeg": 250,   # 1 second at 250Hz
            "gsr": 20,    # 2 seconds at 10Hz
            "acc": 50     # 0.5 seconds at 100Hz
        },
        feature_dims: Dict[str, int] = {
            "ecg": 3,     # Raw, filtered, derivative
            "emg": 4,     # Raw, filtered, RMS, frequency
            "ppg": 3,     # Raw, filtered, pulse rate
            "eeg": 5,     # Raw + 4 frequency bands
            "gsr": 2,     # Raw, tonic component
            "acc": 3      # X, Y, Z axes
        }
    ):
        """
        Initialize the transformer analysis module
        
        Args:
            model_dir: Directory to store/load models
            device: Device to run models on ("cuda" or "cpu")
            signal_types: List of bio-signal types to analyze
            window_sizes: Dictionary mapping signal types to window sizes
            feature_dims: Dictionary mapping signal types to feature dimensions
        """
        self.model_dir = model_dir
        self.device = device
        self.signal_types = signal_types
        self.window_sizes = window_sizes
        self.feature_dims = feature_dims
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models for each signal type
        self.models = {}
        self.data_buffers = {}
        self.last_analysis_time = {}
        self.analysis_results = {}
        
        # Initialize models
        self._initialize_models()
        
        # Analysis settings
        self.real_time_analysis = True
        self.analysis_interval_ms = 500  # Analyze every 500ms
        
        logger.info(f"Transformer analysis module initialized on {device}")
        logger.info(f"Monitoring signal types: {', '.join(signal_types)}")
    
    def _initialize_models(self):
        """
        Initialize transformer models for each signal type
        
        This method either loads existing models or creates new ones
        """
        for signal_type in self.signal_types:
            # Initialize data buffer for this signal type
            self.data_buffers[signal_type] = {
                'data': np.zeros((0, self.feature_dims[signal_type])),
                'timestamps': []
            }
            
            # Initialize last analysis time
            self.last_analysis_time[signal_type] = 0
            
            # Initialize analysis results
            self.analysis_results[signal_type] = {
                'anomaly_scores': [],
                'anomaly_timestamps': [],
                'reconstruction_errors': [],
                'detected_anomalies': []
            }
            
            # Set model path
            model_path = os.path.join(self.model_dir, f"{signal_type}_anomaly_detector.pt")
            
            # Try to load model if it exists
            if os.path.exists(model_path):
                try:
                    self.models[signal_type] = BioSignalAnomalyDetector.load_model(
                        path=model_path,
                        device=self.device
                    )
                    logger.info(f"Loaded existing model for {signal_type}")
                except Exception as e:
                    logger.error(f"Error loading model for {signal_type}: {e}")
                    self._create_new_model(signal_type)
            else:
                # Create new model
                self._create_new_model(signal_type)
    
    def _create_new_model(self, signal_type: str):
        """
        Create a new transformer model for a signal type
        
        Args:
            signal_type: Signal type to create model for
        """
        logger.info(f"Creating new model for {signal_type}")
        
        # Get configuration for this signal type
        window_size = self.window_sizes[signal_type]
        feature_dim = self.feature_dims[signal_type]
        
        # Create model with appropriate parameters based on signal type
        if signal_type == "eeg":
            # EEG requires more complex model due to spatial relationships
            self.models[signal_type] = BioSignalAnomalyDetector(
                input_dim=feature_dim,
                d_model=128,
                num_heads=8,
                d_ff=512,
                num_layers=4,
                max_seq_length=window_size,
                dropout=0.2,
                device=self.device
            )
        elif signal_type in ["ecg", "emg"]:
            # ECG and EMG need moderate complexity
            self.models[signal_type] = BioSignalAnomalyDetector(
                input_dim=feature_dim,
                d_model=64,
                num_heads=4,
                d_ff=256,
                num_layers=3,
                max_seq_length=window_size,
                dropout=0.1,
                device=self.device
            )
        else:
            # Other signals can use simpler models
            self.models[signal_type] = BioSignalAnomalyDetector(
                input_dim=feature_dim,
                d_model=32,
                num_heads=2,
                d_ff=128,
                num_layers=2,
                max_seq_length=window_size,
                dropout=0.1,
                device=self.device
            )
        
        logger.info(f"Created model for {signal_type} with window size {window_size}")
    
    def add_data(self, signal_type: str, data: np.ndarray, timestamp: Optional[float] = None):
        """
        Add new data for a specific signal type
        
        Args:
            signal_type: Signal type to add data for
            data: Data array with shape [samples, features]
            timestamp: Timestamp for the data (defaults to current time)
        """
        if signal_type not in self.signal_types:
            logger.warning(f"Unknown signal type: {signal_type}")
            return
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure data has correct shape
        if len(data.shape) == 1:
            # Single sample, reshape
            data = data.reshape(1, -1)
        
        # Check feature dimension
        if data.shape[1] != self.feature_dims[signal_type]:
            logger.warning(f"Data for {signal_type} has incorrect feature dimension: "
                          f"{data.shape[1]} (expected {self.feature_dims[signal_type]})")
            return
        
        # Add data to buffer
        self.data_buffers[signal_type]['data'] = np.vstack((
            self.data_buffers[signal_type]['data'],
            data
        ))
        
        # Add timestamps (one per sample)
        if isinstance(timestamp, (int, float)):
            # Single timestamp, replicate for each sample
            timestamps = [timestamp + i/1000 for i in range(data.shape[0])]
        else:
            # List of timestamps
            timestamps = timestamp
        
        self.data_buffers[signal_type]['timestamps'].extend(timestamps)
        
        # Trim buffer if it exceeds window size
        window_size = self.window_sizes[signal_type]
        if self.data_buffers[signal_type]['data'].shape[0] > window_size * 2:
            # Keep the most recent window_size*2 samples
            self.data_buffers[signal_type]['data'] = self.data_buffers[signal_type]['data'][-window_size*2:]
            self.data_buffers[signal_type]['timestamps'] = self.data_buffers[signal_type]['timestamps'][-window_size*2:]
        
        # Perform real-time analysis if enabled
        if self.real_time_analysis:
            current_time = time.time()
            if current_time - self.last_analysis_time[signal_type] >= self.analysis_interval_ms / 1000:
                self.analyze_signal(signal_type)
                self.last_analysis_time[signal_type] = current_time
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data for model input
        
        Args:
            data: Data array with shape [samples, features]
            
        Returns:
            Normalized data array
        """
        # Skip if data is empty
        if data.size == 0:
            return data
            
        # Simple z-score normalization with clipping to handle outliers
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        
        normalized = (data - mean) / std
        
        # Clip extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def _assess_data_quality(self, data: np.ndarray) -> float:
        """
        Assess the quality of the data
        
        Args:
            data: Data array with shape [samples, features]
            
        Returns:
            Quality score between 0 and 1
        """
        if data.size == 0:
            return 0.0
            
        # Check for NaN or infinite values
        invalid_ratio = np.isnan(data).sum() + np.isinf(data).sum()
        invalid_ratio = invalid_ratio / data.size
        
        # Check for constant values (zero variance)
        var = np.var(data, axis=0)
        constant_ratio = np.sum(var < 1e-10) / data.shape[1]
        
        # Check for outliers (beyond 5 standard deviations)
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        z_scores = np.abs((data - mean) / std)
        outlier_ratio = np.sum(z_scores > 5.0) / data.size
        
        # Calculate quality score (1.0 is best, 0.0 is worst)
        quality_score = 1.0 - (invalid_ratio + constant_ratio + outlier_ratio)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return float(quality_score)
    
    def analyze_signal(self, signal_type: str) -> Dict[str, Any]:
        """
        Analyze data for a specific signal type
        
        Args:
            signal_type: Signal type to analyze
        
        Returns:
            Dictionary with analysis results
        """
        if signal_type not in self.signal_types:
            logger.warning(f"Unknown signal type: {signal_type}")
            return {}
        
        # Get data from buffer
        data = self.data_buffers[signal_type]['data']
        timestamps = self.data_buffers[signal_type]['timestamps']
        
        # Check if we have enough data
        window_size = self.window_sizes[signal_type]
        if data.shape[0] < window_size:
            logger.debug(f"Not enough data for {signal_type}: {data.shape[0]}/{window_size}")
            return {}
        
        # Prepare data for analysis
        # Use the most recent window_size samples
        analysis_data = data[-window_size:].copy()
        analysis_timestamps = timestamps[-window_size:]
        
        # Normalize data (important for transformer input)
        analysis_data_norm = self._normalize_data(analysis_data)
        
        # Convert to tensor
        x = torch.tensor(analysis_data_norm, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        if x.device != self.device:
            x = x.to(self.device)
        
        # Run anomaly detection with details
        try:
            anomalies, details = self.models[signal_type].detect_anomalies(
                x=x,
                return_details=True
            )
            
            # Process results
            anomaly_scores = details['anomaly_scores'][0]  # Remove batch dimension
            reconstruction_error = details['reconstruction_error'][0]  # Remove batch dimension
            detected_anomalies = anomalies[0].cpu().numpy()  # Remove batch dimension
            
            # Store results
            self.analysis_results[signal_type]['anomaly_scores'].append(anomaly_scores.mean())
            self.analysis_results[signal_type]['anomaly_timestamps'].append(time.time())
            self.analysis_results[signal_type]['reconstruction_errors'].append(reconstruction_error.mean())
            self.analysis_results[signal_type]['detected_anomalies'].append(np.any(detected_anomalies))
            
            # Trim result history if too long
            max_history = 1000
            if len(self.analysis_results[signal_type]['anomaly_scores']) > max_history:
                self.analysis_results[signal_type]['anomaly_scores'] = self.analysis_results[signal_type]['anomaly_scores'][-max_history:]
                self.analysis_results[signal_type]['anomaly_timestamps'] = self.analysis_results[signal_type]['anomaly_timestamps'][-max_history:]
                self.analysis_results[signal_type]['reconstruction_errors'] = self.analysis_results[signal_type]['reconstruction_errors'][-max_history:]
                self.analysis_results[signal_type]['detected_anomalies'] = self.analysis_results[signal_type]['detected_anomalies'][-max_history:]
            
            # Calculate data quality
            data_quality = self._assess_data_quality(analysis_data)
            
            # Prepare results
            results = {
                'signal_type': signal_type,
                'timestamp': time.time(),
                'window_timestamps': analysis_timestamps,
                'anomaly_detected': np.any(detected_anomalies),
                'anomaly_score': float(anomaly_scores.mean()),
                'anomaly_threshold': float(self.models[signal_type].anomaly_threshold),
                'reconstruction_error': float(reconstruction_error.mean()),
                'anomaly_positions': np.where(detected_anomalies)[0].tolist(),
                'data_quality': data_quality,
                'feature_importance': self._calculate_feature_importance(
                    signal_type, analysis_data, detected_anomalies
                )
            }
            
            logger.debug(f"Analysis results for {signal_type}: "
                       f"anomaly={results['anomaly_detected']}, "
                       f"score={results['anomaly_score']:.4f}, "
                       f"quality={results['data_quality']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {signal_type}: {e}")
            return {}
    
    def _calculate_feature_importance(
        self, 
        signal_type: str, 
        data: np.ndarray, 
        anomalies: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate importance of each feature in anomaly detection
        
        Args:
            signal_type: Signal type
            data: Data array with shape [samples, features]
            anomalies: Boolean array indicating anomalies
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Skip if no anomalies detected
        if not np.any(anomalies):
            return {}
            
        # Get feature names based on signal type
        if signal_type == "ecg":
            feature_names = ["Raw", "Filtered", "Derivative"]
        elif signal_type == "emg":
            feature_names = ["Raw", "Filtered", "RMS", "Frequency"]
        elif signal_type == "ppg":
            feature_names = ["Raw", "Filtered", "Pulse Rate"]
        elif signal_type == "eeg":
            feature_names = ["Raw", "Delta", "Theta", "Alpha", "Beta"]
        elif signal_type == "gsr":
            feature_names = ["Raw", "Tonic"]
        elif signal_type == "acc":
            feature_names = ["X", "Y", "Z"]
        else:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        
        # Ensure we have the right number of feature names
        feature_names = feature_names[:data.shape[1]]
        while len(feature_names) < data.shape[1]:
            feature_names.append(f"Feature {len(feature_names)+1}")
        
        # Calculate importance based on difference between normal and anomalous data
        normal_indices = ~anomalies
        anomaly_indices = anomalies
        
        if not np.any(normal_indices) or not np.any(anomaly_indices):
            # Fallback if all samples are normal or all are anomalous
            return {name: 1.0/len(feature_names) for name in feature_names}
        
        normal_data = data[normal_indices]
        anomaly_data = data[anomaly_indices]
        
        # Calculate mean and std for normal data
        normal_mean = np.mean(normal_data, axis=0)
        normal_std = np.std(normal_data, axis=0)
        normal_std = np.where(normal_std < 1e-10, 1.0, normal_std)  # Avoid division by zero
        
        # Calculate mean for anomaly data
        anomaly_mean = np.mean(anomaly_data, axis=0)
        
        # Calculate z-score difference between normal and anomalous data
        z_diff = np.abs((anomaly_mean - normal_mean) / normal_std)
        
        # Normalize importance scores
        importance_sum = np.sum(z_diff)
        if importance_sum > 0:
            importance_scores = z_diff / importance_sum
        else:
            importance_scores = np.ones_like(z_diff) / len(z_diff)
        
        # Create feature importance dictionary
        feature_importance = {
            name: float(score) for name, score in zip(feature_names, importance_scores)
        }
        
        return feature_importance
    
    def train_model(
        self, 
        signal_type: str, 
        training_data: np.ndarray, 
        validation_data: Optional[np.ndarray] = None,
        anomaly_labels: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_model: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train or update a model for a specific signal type
        
        Args:
            signal_type: Signal type to train model for
            training_data: Training data with shape [num_samples, window_size, features]
            validation_data: Validation data with shape [num_samples, window_size, features]
            anomaly_labels: Optional anomaly labels with shape [num_samples, window_size]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_model: Whether to save the model after training
            
        Returns:
            Training history
        """
        if signal_type not in self.signal_types:
            logger.warning(f"Unknown signal type: {signal_type}")
            return {}
        
        logger.info(f"Training model for {signal_type} with {training_data.shape[0]} samples")
        
        # Normalize training data
        training_data_norm = np.array([
            self._normalize_data(sample) for sample in training_data
        ])
        
        # Normalize validation data if provided
        validation_data_norm = None
        if validation_data is not None:
            validation_data_norm = np.array([
                self._normalize_data(sample) for sample in validation_data
            ])
        
        # Convert to tensors
        train_tensor = torch.tensor(training_data_norm, dtype=torch.float32)
        val_tensor = torch.tensor(validation_data_norm, dtype=torch.float32) if validation_data_norm is not None else None
        anomaly_tensor = torch.tensor(anomaly_labels, dtype=torch.float32) if anomaly_labels is not None else None
        
        # Train model
        history = self.models[signal_type].fit(
            train_data=train_tensor,
            val_data=val_tensor,
            anomaly_labels=anomaly_tensor,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Calibrate threshold
        if validation_data_norm is not None:
            logger.info(f"Calibrating threshold for {signal_type}")
            threshold = self.models[signal_type].calibrate_threshold(
                normal_data=val_tensor,
                percentile=95.0
            )
            logger.info(f"Calibrated threshold for {signal_type}: {threshold:.6f}")
        
        # Save model if requested
        if save_model:
            model_path = os.path.join(self.model_dir, f"{signal_type}_anomaly_detector.pt")
            self.models[signal_type].save_model(model_path)
        
        return history
    
    def get_recent_analysis(self, signal_type: str, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Get recent analysis results for a specific signal type
        
        Args:
            signal_type: Signal type to get results for
            num_samples: Number of most recent samples to return
            
        Returns:
            Dictionary with analysis history
        """
        if signal_type not in self.signal_types:
            logger.warning(f"Unknown signal type: {signal_type}")
            return {}
        
        # Get recent results
        results = self.analysis_results[signal_type]
        
        # Limit to requested number of samples
        recent_results = {
            'anomaly_scores': np.array(results['anomaly_scores'][-num_samples:]),
            'anomaly_timestamps': np.array(results['anomaly_timestamps'][-num_samples:]),
            'reconstruction_errors': np.array(results['reconstruction_errors'][-num_samples:]),
            'detected_anomalies': np.array(results['detected_anomalies'][-num_samples:])
        }
        
        return recent_results
    
    def get_stimulation_parameters(
        self, 
        signal_type: str,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimal stimulation parameters based on analysis results
        
        Args:
            signal_type: Signal type to get parameters for
            base_params: Base stimulation parameters to adjust
            
        Returns:
            Adjusted stimulation parameters
        """
        if signal_type not in self.signal_types:
            logger.warning(f"Unknown signal type: {signal_type}")
            return base_params.copy()
        
        # Get recent analysis results
        recent_results = self.get_recent_analysis(signal_type, num_samples=10)
        
        # Copy base parameters
        adjusted_params = base_params.copy()
        
        # Check if we have enough data
        if len(recent_results.get('anomaly_scores', [])) < 5:
            logger.debug(f"Not enough analysis results for {signal_type}")
            return adjusted_params
        
        # Get latest anomaly information
        anomaly_detected = any(recent_results['detected_anomalies'][-3:])  # Any anomalies in last 3 samples
        anomaly_score = float(np.mean(recent_results['anomaly_scores'][-3:]))  # Average over last 3 samples
        anomaly_trend = np.polyfit(
            np.arange(len(recent_results['anomaly_scores'][-10:])),
            recent_results['anomaly_scores'][-10:],
            1
        )[0]  # Slope of anomaly scores over last 10 samples
        
        # Adjust parameters based on anomaly information
        if signal_type == "ecg":
            self._adjust_ecg_stimulation(adjusted_params, anomaly_detected, anomaly_score, anomaly_trend)
        elif signal_type == "emg":
            self._adjust_emg_stimulation(adjusted_params, anomaly_detected, anomaly_score, anomaly_trend)
        elif signal_type == "ppg":
            self._adjust_ppg_stimulation(adjusted_params, anomaly_detected, anomaly_score, anomaly_trend)
        elif signal_type == "eeg":
            self._adjust_eeg_stimulation(adjusted_params, anomaly_detected, anomaly_score, anomaly_trend)
        elif signal_type == "gsr":
            self._adjust_gsr_stimulation(adjusted_params, anomaly_detected, anomaly_score, anomaly_trend)
        
        return adjusted_params
    
    def _adjust_ecg_stimulation(
        self, 
        params: Dict[str, Any],
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_trend: float
    ):
        """
        Adjust stimulation parameters for ECG anomalies
        
        Args:
            params: Parameters to adjust (modified in-place)
            anomaly_detected: Whether anomalies were detected
            anomaly_score: Anomaly score
            anomaly_trend: Trend in anomaly scores
        """
        # Adjust intensity based on anomaly score
        if 'intensity' in params:
            if anomaly_detected:
                # Reduce intensity if anomalies detected (could be arrhythmia)
                params['intensity'] = max(5, params['intensity'] * 0.8)
            elif anomaly_trend < -0.05:
                # Gradually increase intensity if anomaly scores are decreasing
                params['intensity'] = min(params['intensity'] * 1.1, params.get('max_intensity', 50))
        
        # Adjust frequency based on anomaly type
        if 'frequency' in params and anomaly_detected:
            # Lower frequency for ECG anomalies to avoid interference with heart rhythm
            params['frequency'] = min(params['frequency'], 20)
    
    def _adjust_emg_stimulation(
        self, 
        params: Dict[str, Any],
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_trend: float
    ):
        """
        Adjust stimulation parameters for EMG anomalies
        
        Args:
            params: Parameters to adjust (modified in-place)
            anomaly_detected: Whether anomalies were detected
            anomaly_score: Anomaly score
            anomaly_trend: Trend in anomaly scores
        """
        # Adjust intensity based on anomaly score
        if 'intensity' in params:
            if anomaly_detected:
                # Increase intensity to counteract muscle tension or weakness
                adjustment_factor = 1.0 + min(0.5, anomaly_score / 2)
                params['intensity'] = min(params['intensity'] * adjustment_factor, 
                                          params.get('max_intensity', 50))
        
        # Adjust pulse width based on anomaly type
        if 'pulse_width' in params and anomaly_detected:
            # Increase pulse width for better muscle penetration
            params['pulse_width'] = min(params['pulse_width'] * 1.2, 
                                        params.get('max_pulse_width', 300))
    
    def _adjust_ppg_stimulation(
        self, 
        params: Dict[str, Any],
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_trend: float
    ):
        """
        Adjust stimulation parameters for PPG (blood flow) anomalies
        
        Args:
            params: Parameters to adjust (modified in-place)
            anomaly_detected: Whether anomalies were detected
            anomaly_score: Anomaly score
            anomaly_trend: Trend in anomaly scores
        """
        # Adjust stimulation mode based on anomaly
        if 'mode' in params and anomaly_detected:
            # Switch to vasodilation mode for blood flow issues
            params['mode'] = 'vasodilation'
        
        # Adjust pulse rate based on anomaly trend
        if 'pulse_rate' in params:
            if anomaly_trend > 0.05:
                # Increase pulse rate if anomalies are increasing (poor circulation)
                params['pulse_rate'] = min(params['pulse_rate'] * 1.2, 
                                          params.get('max_pulse_rate', 12))
            elif not anomaly_detected:
                # Return to normal pulse rate if no anomalies
                params['pulse_rate'] = params.get('default_pulse_rate', 8)
    
    def _adjust_eeg_stimulation(
        self, 
        params: Dict[str, Any],
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_trend: float
    ):
        """
        Adjust stimulation parameters for EEG anomalies
        
        Args:
            params: Parameters to adjust (modified in-place)
            anomaly_detected: Whether anomalies were detected
            anomaly_score: Anomaly score
            anomaly_trend: Trend in anomaly scores
        """
        # Adjust frequency based on anomaly
        if 'frequency' in params and anomaly_detected:
            # Target alpha frequency range (8-12 Hz) for relaxation
            params['frequency'] = 10.0
        
        # Adjust modulation parameters based on anomaly score
        if 'modulation_depth' in params:
            if anomaly_detected:
                # Increase modulation depth for stronger neural entrainment
                params['modulation_depth'] = min(0.8, params['modulation_depth'] * 1.2)
            else:
                # Return to normal modulation depth
                params['modulation_depth'] = params.get('default_modulation_depth', 0.5)
    
    def _adjust_gsr_stimulation(
        self, 
        params: Dict[str, Any],
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_trend: float
    ):
        """
        Adjust stimulation parameters for GSR (stress) anomalies
        
        Args:
            params: Parameters to adjust (modified in-place)
            anomaly_detected: Whether anomalies were detected
            anomaly_score: Anomaly score
            anomaly_trend: Trend in anomaly scores
        """
        # For GSR (stress response), adjust relaxation parameters
        if 'intensity' in params and anomaly_detected:
            # Reduce intensity for stress-related GSR anomalies
            params['intensity'] = max(params['intensity'] * 0.75, params.get('min_intensity', 5))
        
        # Adjust stimulation duration based on anomaly trend
        if 'duration' in params:
            if anomaly_trend > 0.05:
                # Increase duration if stress is increasing
                params['duration'] = min(params['duration'] * 1.5, params.get('max_duration', 1800))
            elif not anomaly_detected:
                # Return to normal duration
                params['duration'] = params.get('default_duration', 900)
    
    def visualize_signals(
        self,
        signals: Dict[str, np.ndarray],
        anomaly_results: Optional[Dict[str, Dict[str, Any]]] = None,
        time_axis: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple bio-signals with anomaly detection results
        
        Args:
            signals: Dictionary mapping signal types to data arrays
            anomaly_results: Optional dictionary mapping signal types to anomaly results
            time_axis: Optional time axis for plotting
            save_path: Optional path to save the figure
        """
        # Check if we have signals to plot
        if not signals:
            logger.warning("No signals to visualize")
            return
        
        # Create figure
        num_signals = len(signals)
        fig, axes = plt.subplots(num_signals, 1, figsize=(12, 3 * num_signals), sharex=True)
        
        # Handle single signal case
        if num_signals == 1:
            axes = [axes]
        
        # Create time axis if not provided
        if time_axis is None:
            # Use the longest signal for time axis
            max_length = max(data.shape[0] for data in signals.values())
            time_axis = np.arange(max_length) / 100.0  # Assuming 100 Hz sampling rate
        
        # Plot each signal
        for i, (signal_type, data) in enumerate(signals.items()):
            ax = axes[i]
            
            # Plot signal data
            for j in range(min(3, data.shape[1])):  # Plot up to 3 channels
                if data.shape[1] == 1:
                    label = signal_type
                else:
                    # Get channel name based on signal type
                    if signal_type == "ecg" and j < len(["Raw", "Filtered", "Derivative"]):
                        channel_name = ["Raw", "Filtered", "Derivative"][j]
                    elif signal_type == "emg" and j < len(["Raw", "Filtered", "RMS", "Frequency"]):
                        channel_name = ["Raw", "Filtered", "RMS", "Frequency"][j]
                    elif signal_type == "ppg" and j < len(["Raw", "Filtered", "Pulse Rate"]):
                        channel_name = ["Raw", "Filtered", "Pulse Rate"][j]
                    elif signal_type == "eeg" and j < len(["Raw", "Delta", "Theta", "Alpha", "Beta"]):
                        channel_name = ["Raw", "Delta", "Theta", "Alpha", "Beta"][j]
                    elif signal_type == "gsr" and j < len(["Raw", "Tonic"]):
                        channel_name = ["Raw", "Tonic"][j]
                    elif signal_type == "acc" and j < len(["X", "Y", "Z"]):
                        channel_name = ["X", "Y", "Z"][j]
                    else:
                        channel_name = f"Channel {j+1}"
                    
                    label = f"{signal_type} - {channel_name}"
                
                # Plot the signal channel
                ax.plot(time_axis[:data.shape[0]], data[:, j], label=label)
            
            # Plot anomaly regions if available
            if anomaly_results and signal_type in anomaly_results:
                result = anomaly_results[signal_type]
                if 'anomaly_positions' in result:
                    for pos in result['anomaly_positions']:
                        if pos < data.shape[0]:
                            ax.axvspan(time_axis[pos], time_axis[min(pos+1, data.shape[0]-1)],
                                      alpha=0.3, color='red')
            
            # Add labels and legend
            ax.set_ylabel(signal_type.upper())
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add anomaly score if available
            if anomaly_results and signal_type in anomaly_results:
                result = anomaly_results[signal_type]
                if 'anomaly_score' in result:
                    score = result['anomaly_score']
                    threshold = result.get('anomaly_threshold', 0.5)
                    ax.text(0.02, 0.95, f"Anomaly Score: {score:.4f} (Threshold: {threshold:.4f})",
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set common labels
        axes[-1].set_xlabel('Time (s)')
        
        # Set title
        plt.suptitle('Multi-Modal Bio-Signal Analysis', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or display figure
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_report(
        self,
        signal_data: Dict[str, np.ndarray],
        time_period: Tuple[float, float],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report for multiple signals
        
        Args:
            signal_data: Dictionary mapping signal types to data arrays
            time_period: Tuple of (start_time, end_time) for the report
            save_path: Optional path to save the report figures
            
        Returns:
            Report data dictionary
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'time_period': time_period,
            'signals': {},
            'anomalies': {},
            'recommendations': {}
        }
        
        # Analyze each signal
        anomaly_results = {}
        for signal_type, data in signal_data.items():
            if signal_type not in self.signal_types:
                continue
                
            # Analyze the signal
            logger.info(f"Analyzing {signal_type} for report")
            
            # Add whole signal to buffer temporarily
            original_buffer = self.data_buffers[signal_type]['data'].copy()
            original_timestamps = self.data_buffers[signal_type]['timestamps'].copy()
            
            self.data_buffers[signal_type]['data'] = data
            self.data_buffers[signal_type]['timestamps'] = np.linspace(
                time_period[0], time_period[1], data.shape[0]
            ).tolist()
            
            # Run analysis
            result = self.analyze_signal(signal_type)
            anomaly_results[signal_type] = result
            
            # Restore original buffer
            self.data_buffers[signal_type]['data'] = original_buffer
            self.data_buffers[signal_type]['timestamps'] = original_timestamps
            
            # Store signal statistics
            report['signals'][signal_type] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'data_quality': result.get('data_quality', 0.0),
                'feature_importance': result.get('feature_importance', {})
            }
            
            # Store anomaly information
            if result.get('anomaly_detected', False):
                report['anomalies'][signal_type] = {
                    'score': result.get('anomaly_score', 0.0),
                    'threshold': result.get('anomaly_threshold', 0.5),
                    'positions': result.get('anomaly_positions', []),
                    'timestamp': result.get('timestamp', time.time())
                }
        
        # Generate recommendations based on anomalies
        report['recommendations'] = self._generate_recommendations(report['anomalies'])
        
        # Generate visualization
        if save_path:
            self.visualize_signals(
                signals=signal_data,
                anomaly_results=anomaly_results,
                time_axis=np.linspace(time_period[0], time_period[1], 
                                     max(data.shape[0] for data in signal_data.values())),
                save_path=f"{save_path}_overview.png"
            )
            
            # Generate individual signal visualizations
            for signal_type, data in signal_data.items():
                if signal_type in anomaly_results:
                    if self.models[signal_type] is not None:
                        # Use model's visualization function
                        window_size = self.window_sizes[signal_type]
                        if data.shape[0] >= window_size:
                            # Normalize the data
                            data_norm = self._normalize_data(data[-window_size:])
                            x = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0)
                            if x.device != self.device:
                                x = x.to(self.device)
                                
                            # Visualize anomalies
                            self.models[signal_type].visualize_anomalies(
                                x=x,
                                save_path=f"{save_path}_{signal_type}_anomalies.png"
                            )
        
        return report
    
    def _generate_recommendations(self, anomalies: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate recommendations based on detected anomalies
        
        Args:
            anomalies: Dictionary mapping signal types to anomaly information
            
        Returns:
            Dictionary mapping recommendation types to recommendation text
        """
        recommendations = {}
        
        # Check ECG anomalies
        if 'ecg' in anomalies:
            recommendations['heart_rate'] = (
                "ECG anomalies detected. Consider reducing physical exertion and monitoring "
                "heart rate closely. TENS stimulation parameters have been automatically "
                "adjusted to reduce intensity and frequency."
            )
        
        # Check EMG anomalies
        if 'emg' in anomalies:
            recommendations['muscle_activity'] = (
                "Abnormal muscle activity detected. TENS stimulation has been adjusted to "
                "optimize muscle relaxation. Consider incorporating the suggested stretching "
                "exercises and proper posture techniques."
            )
        
        # Check PPG anomalies
        if 'ppg' in anomalies:
            recommendations['blood_flow'] = (
                "Blood flow irregularities detected. Stimulation parameters have been "
                "adjusted to enhance circulation. Consider elevating affected limbs and "
                "staying well-hydrated."
            )
        
        # Check EEG anomalies
        if 'eeg' in anomalies:
            recommendations['brain_activity'] = (
                "EEG patterns show potential irregularities. Neurostimulation has been "
                "adjusted to target alpha wave enhancement. Recommended activities include "
                "brief meditation sessions and reduced screen time."
            )
        
        # Check GSR anomalies
        if 'gsr' in anomalies:
            recommendations['stress_response'] = (
                "Elevated stress response detected. Stimulation has been adjusted to promote "
                "relaxation. Recommended breathing exercises included in the companion app."
            )
        
        # General recommendation if multiple anomalies detected
        if len(anomalies) >= 2:
            recommendations['system_adjustment'] = (
                "Multiple signal anomalies detected. The system has automatically adjusted "
                "stimulation parameters across all modalities to create a harmonized therapeutic "
                "effect. Monitor responses over the next 24 hours and consider consulting "
                "a healthcare professional if anomalies persist."
            )
        
        # Add default recommendation if no specific ones
        if not recommendations:
            recommendations['general'] = (
                "No significant anomalies detected in the monitored bio-signals. "
                "Current stimulation parameters are maintained for optimal effect. "
                "Continue regular monitoring and therapy sessions as scheduled."
            )
        
        return recommendations


# Example usage when run as a script
if __name__ == "__main__":
    # Create transformer analysis module
    analyzer = TransformerAnalysisModule(
        model_dir="models",
        device="cpu"  # Use CPU for demonstration
    )
    
    # Generate some example data
    import matplotlib.pyplot as plt
    
    # Generate ECG data (200 Hz for 5 seconds)
    t = np.linspace(0, 5, 1000)
    ecg_raw = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = (ti % 0.8) / 0.8 * 2 * np.pi
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
    
    # Add some noise
    ecg_raw += 0.05 * np.random.randn(len(ecg_raw))
    
    # Add an artificial anomaly
    ecg_raw[600:650] += 0.5 * np.sin(np.linspace(0, 4*np.pi, 50))
    
    # Create derivative and filtered versions
    ecg_derivative = np.gradient(ecg_raw)
    ecg_filtered = ecg_raw.copy()
    
    # Combine into multi-channel data
    ecg_data = np.column_stack((ecg_raw, ecg_filtered, ecg_derivative))
    
    # Add to analyzer
    for i in range(0, len(ecg_data), 50):
        chunk = ecg_data[i:i+50]
        if len(chunk) > 0:
            analyzer.add_data("ecg", chunk, timestamp=t[i])
    
    # Analyze
    result = analyzer.analyze_signal("ecg")
    print("ECG Analysis Result:")
    for key, value in result.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 10:
            print(f"  {key}: [Array with {len(value)} elements]")
        else:
            print(f"  {key}: {value}")
    
    # Generate report
    report = analyzer.generate_report(
        signal_data={"ecg": ecg_data},
        time_period=(0, 5),
        save_path="bio_signal_report"
    )
    
    print("\nReport Recommendations:")
    for key, value in report['recommendations'].items():
        print(f"  {key}: {value}")
