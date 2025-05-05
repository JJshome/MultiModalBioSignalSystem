#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-Based Bio-Signal Anomaly Detection Module

This module implements the transformer-based bio-signal anomaly detection system
described in the patent "Multi-Modal Bio-Stimulation, Diagnosis and Treatment System
Using Multiple Wireless Devices and Method for the Same" by UCare-tron Inc.

The system uses a transformer architecture to analyze time-series bio-signal data
(ECG, EMG, EEG, etc.) and detect anomalous patterns that may indicate
physiological issues or conditions requiring intervention.

Key features:
- Multi-head self-attention mechanism for capturing temporal relationships
- Positional encoding for maintaining sequence information
- Anomaly scoring based on reconstruction error and probability distribution
- Explainable results with attention visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transformer_anomaly_detector")

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer models
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding

        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Calculate sine/cosine positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor

        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for transformer models
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for query, key, value, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attention_weights = None  # Store attention weights for visualization
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split tensor into multiple heads

        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Tensor of shape [batch_size, num_heads, seq_length, d_k]
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back to original shape

        Args:
            x: Tensor of shape [batch_size, num_heads, seq_length, d_k]

        Returns:
            Tensor of shape [batch_size, seq_length, d_model]
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention

        Args:
            q: Query tensor [batch_size, seq_length, d_model]
            k: Key tensor [batch_size, seq_length, d_model]
            v: Value tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, seq_length, seq_length]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        q = self.split_heads(self.q_linear(q))  # [batch_size, num_heads, seq_length, d_k]
        k = self.split_heads(self.k_linear(k))  # [batch_size, num_heads, seq_length, d_k]
        v = self.split_heads(self.v_linear(v))  # [batch_size, num_heads, seq_length, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch_size, num_heads, seq_length, seq_length]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_length, d_k]
        
        # Combine heads and apply final linear layer
        context = self.combine_heads(context)  # [batch_size, seq_length, d_model]
        output = self.output_linear(context)  # [batch_size, seq_length, d_model]
        
        return output

class FeedForward(nn.Module):
    """
    Feed-forward network for transformer models
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feed-forward network

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize encoder layer

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute encoder layer

        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, seq_length, seq_length]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Self-attention with residual connection and normalization
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer encoder
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, 
                 max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize transformer encoder

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of encoder layers
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super(TransformerEncoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute transformer encoder

        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, seq_length, seq_length]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x

class BioSignalAnomalyDetector(nn.Module):
    """
    Transformer-based bio-signal anomaly detector
    
    This model uses a transformer encoder to learn patterns in bio-signal data
    and detect anomalies based on reconstruction error and probability distribution.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_seq_length: int = 1000,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize bio-signal anomaly detector

        Args:
            input_dim: Dimension of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of encoder layers
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            device: Device to run model on
        """
        super(BioSignalAnomalyDetector, self).__init__()
        
        self.device = device
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Feature extraction head
        self.feature_extraction = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Reconstruction head (decoder)
        self.reconstruction = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )
        
        # Anomaly detection head
        self.anomaly_detection = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(device)
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'anomaly_loss': []
        }
        
        # Anomaly detection state
        self.anomaly_threshold = 0.5  # Default threshold
        self.anomaly_history = []
        
    def _init_weights(self):
        """
        Initialize weights for all linear layers
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            x: Input tensor [batch_size, seq_length, input_dim]
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary containing output tensors:
                - reconstructed: Reconstructed signal [batch_size, seq_length, input_dim]
                - anomaly_score: Anomaly score [batch_size, seq_length, 1]
                - features: Feature embeddings [batch_size, seq_length, d_model//4] (if return_embeddings=True)
                - encoded: Encoder output [batch_size, seq_length, d_model] (if return_embeddings=True)
        """
        # Project input to model dimension
        x_projected = self.input_projection(x)  # [batch_size, seq_length, d_model]
        
        # Encode with transformer
        encoded = self.transformer_encoder(x_projected)  # [batch_size, seq_length, d_model]
        
        # Extract features
        features = self.feature_extraction(encoded)  # [batch_size, seq_length, d_model//4]
        
        # Reconstruct input
        reconstructed = self.reconstruction(features)  # [batch_size, seq_length, input_dim]
        
        # Anomaly detection
        anomaly_score = self.anomaly_detection(features)  # [batch_size, seq_length, 1]
        
        # Prepare output
        outputs = {
            'reconstructed': reconstructed,
            'anomaly_score': anomaly_score
        }
        
        if return_embeddings:
            outputs['features'] = features
            outputs['encoded'] = encoded
            
        return outputs
    
    def detect_anomalies(
        self, 
        x: torch.Tensor, 
        threshold: Optional[float] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Detect anomalies in bio-signal data

        Args:
            x: Input tensor [batch_size, seq_length, input_dim]
            threshold: Anomaly detection threshold (default: self.anomaly_threshold)
            return_details: Whether to return detailed anomaly information

        Returns:
            If return_details=False:
                Boolean tensor indicating anomalies [batch_size, seq_length]
            If return_details=True:
                Tuple containing:
                    - Boolean tensor indicating anomalies [batch_size, seq_length]
                    - Dictionary with detailed anomaly information
        """
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.anomaly_threshold
            
        # Ensure model is in evaluation mode
        self.eval()
        
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Move to device if needed
        if x.device != self.device:
            x = x.to(self.device)
            
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(x, return_embeddings=return_details)
            
        # Get anomaly scores
        anomaly_scores = outputs['anomaly_score'].squeeze(-1)  # [batch_size, seq_length]
        
        # Detect anomalies based on threshold
        anomalies = anomaly_scores > threshold  # [batch_size, seq_length]
        
        # Return results
        if not return_details:
            return anomalies
        else:
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(
                outputs['reconstructed'], x, reduction='none'
            ).mean(dim=-1)  # [batch_size, seq_length]
            
            # Calculate attention patterns
            attention_patterns = None
            for layer in self.transformer_encoder.layers:
                if attention_patterns is None:
                    attention_patterns = layer.self_attention.attention_weights
                else:
                    attention_patterns += layer.self_attention.attention_weights
            
            if attention_patterns is not None:
                attention_patterns = attention_patterns / len(self.transformer_encoder.layers)
            
            # Prepare detailed information
            details = {
                'anomaly_scores': anomaly_scores.cpu().numpy(),
                'reconstruction_error': reconstruction_error.cpu().numpy(),
                'attention_patterns': attention_patterns.cpu().numpy() if attention_patterns is not None else None,
                'threshold': threshold,
                'features': outputs['features'].cpu().numpy() if 'features' in outputs else None
            }
            
            return anomalies, details
    
    def fit(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        anomaly_labels: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        anomaly_weight: float = 0.5,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the anomaly detection model

        Args:
            train_data: Training data [num_samples, seq_length, input_dim]
            val_data: Validation data [num_samples, seq_length, input_dim]
            anomaly_labels: Anomaly labels for supervised training [num_samples, seq_length]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            anomaly_weight: Weight for anomaly detection loss
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        # Convert to tensors if needed
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.tensor(train_data, dtype=torch.float32)
            
        if val_data is not None and not isinstance(val_data, torch.Tensor):
            val_data = torch.tensor(val_data, dtype=torch.float32)
            
        if anomaly_labels is not None and not isinstance(anomaly_labels, torch.Tensor):
            anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.float32)
            
        # Move to device
        train_data = train_data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        if anomaly_labels is not None:
            anomaly_labels = anomaly_labels.to(self.device)
            
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=verbose
        )
        
        # Reset training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'anomaly_loss': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            reconstruction_losses = []
            anomaly_losses = []
            
            # Create batches
            num_samples = train_data.size(0)
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                batch_x = train_data[batch_indices]
                batch_anomaly = None
                if anomaly_labels is not None:
                    batch_anomaly = anomaly_labels[batch_indices]
                
                # Forward pass
                outputs = self.forward(batch_x)
                
                # Calculate reconstruction loss
                recon_loss = F.mse_loss(outputs['reconstructed'], batch_x)
                
                # Calculate anomaly detection loss (if labels provided)
                anomaly_loss = 0.0
                if batch_anomaly is not None:
                    anomaly_loss = F.binary_cross_entropy(
                        outputs['anomaly_score'].squeeze(-1), 
                        batch_anomaly
                    )
                
                # Calculate total loss
                loss = recon_loss
                if batch_anomaly is not None:
                    loss += anomaly_weight * anomaly_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record losses
                train_losses.append(loss.item())
                reconstruction_losses.append(recon_loss.item())
                if batch_anomaly is not None:
                    anomaly_losses.append(anomaly_loss.item())
            
            # Calculate mean losses
            mean_train_loss = sum(train_losses) / len(train_losses)
            mean_recon_loss = sum(reconstruction_losses) / len(reconstruction_losses)
            mean_anomaly_loss = sum(anomaly_losses) / len(anomaly_losses) if anomaly_losses else 0.0
            
            # Record training history
            self.training_history['train_loss'].append(mean_train_loss)
            self.training_history['reconstruction_loss'].append(mean_recon_loss)
            self.training_history['anomaly_loss'].append(mean_anomaly_loss)
            
            # Validation
            if val_data is not None:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    # Create batches
                    num_val_samples = val_data.size(0)
                    
                    for start_idx in range(0, num_val_samples, batch_size):
                        # Get batch data
                        batch_x = val_data[start_idx:start_idx + batch_size]
                        batch_anomaly = None
                        if anomaly_labels is not None:
                            batch_anomaly = anomaly_labels[start_idx:start_idx + batch_size]
                        
                        # Forward pass
                        outputs = self.forward(batch_x)
                        
                        # Calculate reconstruction loss
                        recon_loss = F.mse_loss(outputs['reconstructed'], batch_x)
                        
                        # Calculate anomaly detection loss (if labels provided)
                        anomaly_loss = 0.0
                        if batch_anomaly is not None:
                            anomaly_loss = F.binary_cross_entropy(
                                outputs['anomaly_score'].squeeze(-1), 
                                batch_anomaly
                            )
                        
                        # Calculate total loss
                        loss = recon_loss
                        if batch_anomaly is not None:
                            loss += anomaly_weight * anomaly_loss
                        
                        # Record losses
                        val_losses.append(loss.item())
                
                # Calculate mean validation loss
                mean_val_loss = sum(val_losses) / len(val_losses)
                
                # Record validation history
                self.training_history['val_loss'].append(mean_val_loss)
                
                # Update learning rate
                scheduler.step(mean_val_loss)
                
                # Early stopping
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_model_state = self.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if verbose and (epoch + 1) % (max(1, epochs // 10)) == 0:
                val_status = f", val_loss: {mean_val_loss:.6f}" if val_data is not None else ""
                anomaly_status = f", anomaly_loss: {mean_anomaly_loss:.6f}" if anomaly_losses else ""
                print(f"Epoch {epoch+1}/{epochs}, train_loss: {mean_train_loss:.6f}, "
                      f"recon_loss: {mean_recon_loss:.6f}{anomaly_status}{val_status}")
        
        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            
        return self.training_history
    
    def calibrate_threshold(
        self,
        normal_data: torch.Tensor,
        percentile: float = 95.0,
        batch_size: int = 32
    ) -> float:
        """
        Calibrate anomaly detection threshold using normal data

        Args:
            normal_data: Normal (non-anomalous) data [num_samples, seq_length, input_dim]
            percentile: Percentile of anomaly scores to use as threshold
            batch_size: Batch size for processing

        Returns:
            Calibrated anomaly threshold
        """
        # Convert to tensor if needed
        if not isinstance(normal_data, torch.Tensor):
            normal_data = torch.tensor(normal_data, dtype=torch.float32)
            
        # Move to device
        normal_data = normal_data.to(self.device)
        
        # Ensure model is in evaluation mode
        self.eval()
        
        # Collect anomaly scores
        all_scores = []
        
        with torch.no_grad():
            # Process in batches
            num_samples = normal_data.size(0)
            
            for start_idx in range(0, num_samples, batch_size):
                # Get batch data
                batch_x = normal_data[start_idx:min(start_idx + batch_size, num_samples)]
                
                # Forward pass
                outputs = self.forward(batch_x)
                
                # Get anomaly scores
                anomaly_scores = outputs['anomaly_score'].squeeze(-1)  # [batch_size, seq_length]
                
                # Add to collection
                all_scores.append(anomaly_scores)
        
        # Concatenate all scores
        all_scores = torch.cat(all_scores, dim=0)  # [total_samples, seq_length]
        
        # Flatten scores
        all_scores = all_scores.flatten().cpu().numpy()
        
        # Calculate threshold based on percentile
        threshold = np.percentile(all_scores, percentile)
        
        # Update model threshold
        self.anomaly_threshold = threshold
        
        return threshold
    
    def save_model(self, path: str):
        """
        Save model to file

        Args:
            path: Path to save model
        """
        # Create save dictionary
        save_dict = {
            'model_state': self.state_dict(),
            'config': {
                'input_dim': self.input_projection.in_features,
                'd_model': self.d_model,
                'max_seq_length': self.max_seq_length,
                'anomaly_threshold': self.anomaly_threshold
            },
            'training_history': self.training_history,
            'anomaly_history': self.anomaly_history
        }
        
        # Save to file
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load model from file

        Args:
            path: Path to load model from
            device: Device to load model on

        Returns:
            Loaded model
        """
        # Load save dictionary
        save_dict = torch.load(path, map_location=device)
        
        # Get configuration
        config = save_dict['config']
        
        # Create model
        model = cls(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            max_seq_length=config['max_seq_length'],
            device=device
        )
        
        # Load state
        model.load_state_dict(save_dict['model_state'])
        
        # Set threshold
        model.anomaly_threshold = config['anomaly_threshold']
        
        # Load history
        model.training_history = save_dict['training_history']
        model.anomaly_history = save_dict['anomaly_history']
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def visualize_anomalies(
        self,
        x: torch.Tensor,
        time_axis: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize detected anomalies

        Args:
            x: Input data [batch_size, seq_length, input_dim]
            time_axis: Time axis for plotting (default: sample indices)
            feature_names: Names of input features (default: feature indices)
            figsize: Figure size
            save_path: Path to save figure (if None, figure is displayed)
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Move to device if needed
        if x.device != self.device:
            x = x.to(self.device)
            
        # Create time axis if not provided
        if time_axis is None:
            time_axis = np.arange(x.size(1))
            
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(x.size(2))]
            
        # Forward pass with details
        with torch.no_grad():
            anomalies, details = self.detect_anomalies(x, return_details=True)
            
        # Convert to numpy
        input_data = x.cpu().numpy()
        reconstructed = details['reconstruction_error']
        anomaly_scores = details['anomaly_scores']
        
        # Ensure batch dimension is present
        if len(input_data.shape) == 2:
            input_data = input_data[np.newaxis, ...]
            reconstructed = reconstructed[np.newaxis, ...]
            anomaly_scores = anomaly_scores[np.newaxis, ...]
            anomalies = anomalies[np.newaxis, ...]
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Set up subplot grid
        grid_height = min(4, x.size(2) + 2)  # Input features + anomaly score + reconstruction error
        gs = plt.GridSpec(grid_height, 1, height_ratios=[1] * grid_height)
        
        # Plot anomaly scores
        ax_anomaly = fig.add_subplot(gs[0, 0])
        ax_anomaly.plot(time_axis, anomaly_scores[0], 'r-', label='Anomaly Score')
        ax_anomaly.axhline(y=self.anomaly_threshold, color='k', linestyle='--', 
                           label=f'Threshold ({self.anomaly_threshold:.3f})')
        
        # Highlight anomalies
        for i in range(len(time_axis)):
            if anomalies[0, i]:
                ax_anomaly.axvspan(time_axis[i], time_axis[i+1] if i < len(time_axis)-1 else time_axis[i],
                                  alpha=0.3, color='red')
                
        ax_anomaly.set_title('Anomaly Detection')
        ax_anomaly.set_ylabel('Anomaly Score')
        ax_anomaly.legend()
        ax_anomaly.grid(True)
        
        # Plot reconstruction error
        ax_recon = fig.add_subplot(gs[1, 0])
        ax_recon.plot(time_axis, reconstructed[0], 'b-', label='Reconstruction Error')
        
        # Highlight anomalies
        for i in range(len(time_axis)):
            if anomalies[0, i]:
                ax_recon.axvspan(time_axis[i], time_axis[i+1] if i < len(time_axis)-1 else time_axis[i],
                               alpha=0.3, color='red')
                
        ax_recon.set_title('Reconstruction Error')
        ax_recon.set_ylabel('Error')
        ax_recon.legend()
        ax_recon.grid(True)
        
        # Plot input features
        for i in range(min(x.size(2), grid_height - 2)):
            ax_feature = fig.add_subplot(gs[i+2, 0])
            ax_feature.plot(time_axis, input_data[0, :, i], 'g-', label=f'Input')
            
            # Highlight anomalies
            for j in range(len(time_axis)):
                if anomalies[0, j]:
                    ax_feature.axvspan(time_axis[j], time_axis[j+1] if j < len(time_axis)-1 else time_axis[j],
                                     alpha=0.3, color='red')
                    
            ax_feature.set_title(f'{feature_names[i]}')
            ax_feature.set_ylabel('Value')
            ax_feature.grid(True)
            
            if i == x.size(2) - 1 or i == grid_height - 3:
                ax_feature.set_xlabel('Time')
        
        # Set layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def visualize_attention(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        head_idx: int = 0,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize attention patterns

        Args:
            x: Input data [batch_size, seq_length, input_dim]
            layer_idx: Transformer layer index (-1 for last layer)
            head_idx: Attention head index
            figsize: Figure size
            save_path: Path to save figure (if None, figure is displayed)
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Move to device if needed
        if x.device != self.device:
            x = x.to(self.device)
            
        # Ensure batch dimension is present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Forward pass
        with torch.no_grad():
            self.forward(x)
            
        # Get attention weights
        layer = self.transformer_encoder.layers[layer_idx]
        attention_weights = layer.self_attention.attention_weights
        
        # Get weights for specified head
        head_weights = attention_weights[0, head_idx].cpu().numpy()
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.imshow(head_weights, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel('Sequence Position (Key)')
        plt.ylabel('Sequence Position (Query)')
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def demo():
    """
    Demonstrate the use of the transformer-based anomaly detector
    """
    print("Demonstrating Transformer-Based Bio-Signal Anomaly Detector")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic bio-signal data
    seq_length = 200
    input_dim = 3
    num_samples = 100
    
    # Generate normal data (sine waves with some noise)
    normal_data = np.zeros((num_samples, seq_length, input_dim))
    for i in range(num_samples):
        # First channel: sine wave
        normal_data[i, :, 0] = np.sin(np.linspace(0, 4*np.pi, seq_length))
        # Second channel: cosine wave
        normal_data[i, :, 1] = np.cos(np.linspace(0, 4*np.pi, seq_length))
        # Third channel: combined wave
        normal_data[i, :, 2] = 0.5 * np.sin(np.linspace(0, 8*np.pi, seq_length))
        
        # Add noise
        normal_data[i] += 0.1 * np.random.randn(*normal_data[i].shape)
    
    # Generate anomalous data (normal data with anomalies)
    anomalous_data = normal_data.copy()
    anomaly_labels = np.zeros((num_samples, seq_length))
    
    for i in range(num_samples):
        if i % 5 == 0:  # 20% of samples have anomalies
            # Insert anomaly
            anomaly_start = np.random.randint(50, 150)
            anomaly_length = np.random.randint(10, 30)
            
            # Create anomaly (spike or drop)
            if np.random.rand() > 0.5:
                # Spike
                anomalous_data[i, anomaly_start:anomaly_start+anomaly_length] += 1.0
            else:
                # Drop
                anomalous_data[i, anomaly_start:anomaly_start+anomaly_length] -= 1.0
                
            # Mark anomaly in labels
            anomaly_labels[i, anomaly_start:anomaly_start+anomaly_length] = 1.0
    
    # Split data into train/val/test
    train_data = normal_data[:70]
    val_data = normal_data[70:85]
    test_normal = normal_data[85:]
    test_anomalous = anomalous_data[85:]
    test_labels = anomaly_labels[85:]
    
    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_normal = torch.tensor(test_normal, dtype=torch.float32)
    test_anomalous = torch.tensor(test_anomalous, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    # Create model
    model = BioSignalAnomalyDetector(
        input_dim=input_dim,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_length=seq_length,
        dropout=0.1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-5,
        verbose=True
    )
    
    # Calibrate threshold
    print("Calibrating threshold...")
    threshold = model.calibrate_threshold(
        normal_data=val_data,
        percentile=95.0
    )
    print(f"Calibrated threshold: {threshold:.6f}")
    
    # Evaluate on test data
    print("Evaluating on test data...")
    
    # Normal data
    anomalies_normal, details_normal = model.detect_anomalies(
        test_normal, return_details=True
    )
    false_positive_rate = anomalies_normal.float().mean().item()
    print(f"False positive rate: {false_positive_rate:.6f}")
    
    # Anomalous data
    anomalies_anomalous, details_anomalous = model.detect_anomalies(
        test_anomalous, return_details=True
    )
    
    # Calculate metrics
    true_positive = (anomalies_anomalous & test_labels.bool()).sum().item()
    false_positive = (anomalies_anomalous & ~test_labels.bool()).sum().item()
    true_negative = (~anomalies_anomalous & ~test_labels.bool()).sum().item()
    false_negative = (~anomalies_anomalous & test_labels.bool()).sum().item()
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    
    # Visualize results
    print("Visualizing results...")
    model.visualize_anomalies(
        test_anomalous[0:1],
        feature_names=["Sine Wave", "Cosine Wave", "Combined Wave"]
    )
    
    print("Demo complete!")


if __name__ == "__main__":
    demo()
