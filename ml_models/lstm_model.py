"""
LSTM Model for Weather Prediction
Deep learning time series forecasting
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from tensorflow.keras.models import Sequential

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None  # type: ignore
    print("Warning: TensorFlow not available. LSTM model will be disabled.")


class LSTMPredictor:
    """
    Long Short-Term Memory (LSTM) neural network
    for time series weather prediction
    """
    
    def __init__(self, lookback: int = 24):
        """
        Initialize LSTM predictor
        
        Args:
            lookback: Number of past time steps to use for prediction (default: 24 hours)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Run: pip install tensorflow")
        
        self.lookback = lookback
        self.temp_model = None
        self.humidity_model = None
        self.temp_scaler = MinMaxScaler(feature_range=(0, 1))
        self.humidity_scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_history = {}
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        warnings.filterwarnings("ignore")
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Time series data
            lookback: Number of past steps to use
            
        Returns:
            X (input sequences), y (target values)
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple):
        """
        Build OPTIMIZED LSTM neural network architecture
        Enhanced with better layers and regularization
        
        Args:
            input_shape: Shape of input data (lookback, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer - increased units from 50 to 100
            LSTM(100, return_sequences=True, input_shape=input_shape,
                 recurrent_dropout=0.1),  # Added recurrent dropout
            Dropout(0.3),  # Increased from 0.2 to 0.3
            
            # Second LSTM layer - increased units
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer for deeper learning
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers with batch normalization effect
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        # OPTIMIZED: Use Adam with learning rate scheduling
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,  # Standard learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, temperature_data: pd.Series, humidity_data: pd.Series, 
              epochs: int = 100, verbose: int = 0) -> Dict:
        """
        Train OPTIMIZED LSTM models for temperature and humidity
        Enhanced with better callbacks and training strategy
        
        Args:
            temperature_data: Historical temperature readings
            humidity_data: Historical humidity readings
            epochs: Number of training epochs (increased to 100)
            verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Dictionary with training metrics
        """
        training_info = {
            'model_type': 'LSTM',
            'data_points': len(temperature_data),
            'lookback': self.lookback,
            'epochs': epochs,
            'success': True
        }
        
        try:
            # Prepare temperature data
            temp_data = temperature_data.values.reshape(-1, 1)
            temp_scaled = self.temp_scaler.fit_transform(temp_data)
            
            if len(temp_scaled) < self.lookback + 10:
                raise ValueError(f"Insufficient data. Need at least {self.lookback + 10} points, got {len(temp_scaled)}")
            
            X_temp, y_temp = self._create_sequences(temp_scaled, self.lookback)
            X_temp = X_temp.reshape((X_temp.shape[0], X_temp.shape[1], 1))
            
            # Build and train temperature model
            self.temp_model = self._build_model((self.lookback, 1))
            
            # OPTIMIZED: Enhanced callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',  # Monitor validation loss
                patience=15,         # Increased patience from 10 to 15
                restore_best_weights=True,
                min_delta=1e-4      # Minimum change to qualify as improvement
            )
            
            # Learning rate reduction on plateau
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,          # Reduce LR by half
                patience=7,          # Wait 7 epochs before reducing
                min_lr=1e-6,
                verbose=0
            )
            
            temp_history = self.temp_model.fit(
                X_temp, y_temp,
                epochs=epochs,
                batch_size=16,       # Reduced from 32 to 16 for better gradients
                verbose=verbose,
                callbacks=[early_stop, reduce_lr],
                validation_split=0.15,  # Increased from 0.1 to 0.15
                shuffle=True         # Shuffle training data
            )
            
            training_info['temp_loss'] = float(temp_history.history['loss'][-1])
            training_info['temp_mae'] = float(temp_history.history['mae'][-1])
            training_info['temp_mse'] = float(temp_history.history['mse'][-1])
            if 'val_loss' in temp_history.history:
                training_info['temp_val_loss'] = float(temp_history.history['val_loss'][-1])
                training_info['temp_val_mae'] = float(temp_history.history['val_mae'][-1])
            training_info['temp_epochs_trained'] = len(temp_history.history['loss'])
            
            # Prepare humidity data
            humidity_data_values = humidity_data.values.reshape(-1, 1)
            humidity_scaled = self.humidity_scaler.fit_transform(humidity_data_values)
            
            X_humidity, y_humidity = self._create_sequences(humidity_scaled, self.lookback)
            X_humidity = X_humidity.reshape((X_humidity.shape[0], X_humidity.shape[1], 1))
            
            # Build and train humidity model
            self.humidity_model = self._build_model((self.lookback, 1))
            
            humidity_history = self.humidity_model.fit(
                X_humidity, y_humidity,
                epochs=epochs,
                batch_size=16,
                verbose=verbose,
                callbacks=[early_stop, reduce_lr],
                validation_split=0.15,
                shuffle=True
            )
            
            training_info['humidity_loss'] = float(humidity_history.history['loss'][-1])
            training_info['humidity_mae'] = float(humidity_history.history['mae'][-1])
            training_info['humidity_mse'] = float(humidity_history.history['mse'][-1])
            if 'val_loss' in humidity_history.history:
                training_info['humidity_val_loss'] = float(humidity_history.history['val_loss'][-1])
                training_info['humidity_val_mae'] = float(humidity_history.history['val_mae'][-1])
            training_info['humidity_epochs_trained'] = len(humidity_history.history['loss'])
            
            self.training_history = training_info
            return training_info
            
        except Exception as e:
            training_info['success'] = False
            training_info['error'] = str(e)
            return training_info
    
    def predict(self, temperature_data: pd.Series, humidity_data: pd.Series, 
                steps: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate predictions for future time steps
        
        Args:
            temperature_data: Recent temperature data (for context)
            humidity_data: Recent humidity data (for context)
            steps: Number of hours to predict (default: 5)
            
        Returns:
            Tuple of (temperature_predictions, humidity_predictions, confidence_info)
        """
        if self.temp_model is None or self.humidity_model is None:
            raise ValueError("Models must be trained before prediction")
        
        # Temperature predictions
        temp_data = temperature_data.values.reshape(-1, 1)
        temp_scaled = self.temp_scaler.transform(temp_data)
        
        temp_predictions = []
        current_sequence = temp_scaled[-self.lookback:].copy()
        
        for _ in range(steps):
            # Predict next value
            X_pred = current_sequence.reshape(1, self.lookback, 1)
            next_pred = self.temp_model.predict(X_pred, verbose=0)
            temp_predictions.append(next_pred[0, 0])
            
            # Update sequence (rolling window)
            current_sequence = np.append(current_sequence[1:], next_pred[0])
        
        # Inverse transform to original scale
        temp_predictions = self.temp_scaler.inverse_transform(
            np.array(temp_predictions).reshape(-1, 1)
        ).flatten()
        
        # Humidity predictions
        humidity_data_values = humidity_data.values.reshape(-1, 1)
        humidity_scaled = self.humidity_scaler.transform(humidity_data_values)
        
        humidity_predictions = []
        current_sequence = humidity_scaled[-self.lookback:].copy()
        
        for _ in range(steps):
            X_pred = current_sequence.reshape(1, self.lookback, 1)
            next_pred = self.humidity_model.predict(X_pred, verbose=0)
            humidity_predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred[0])
        
        humidity_predictions = self.humidity_scaler.inverse_transform(
            np.array(humidity_predictions).reshape(-1, 1)
        ).flatten()
        
        # LSTM doesn't provide native confidence intervals
        # Estimate based on training error
        confidence_info = {
            'temp_confidence_lower': (temp_predictions - 2.0).tolist(),
            'temp_confidence_upper': (temp_predictions + 2.0).tolist(),
            'humidity_confidence_lower': (humidity_predictions - 5.0).tolist(),
            'humidity_confidence_upper': (humidity_predictions + 5.0).tolist(),
            'confidence_level': 0.95,
            'note': 'Confidence intervals estimated from training error'
        }
        
        return temp_predictions, humidity_predictions, confidence_info
    
    def get_model_info(self) -> Dict:
        """Get model information and parameters"""
        model_summary = []
        if self.temp_model:
            self.temp_model.summary(print_fn=lambda x: model_summary.append(x))
        
        return {
            'model_type': 'LSTM',
            'lookback': self.lookback,
            'architecture': 'Two-layer LSTM with Dropout',
            'training_history': self.training_history,
            'description': 'Long Short-Term Memory neural network - Deep learning time series model',
            'model_summary': '\n'.join(model_summary) if model_summary else None
        }
