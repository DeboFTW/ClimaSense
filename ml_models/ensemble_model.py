"""
Ensemble Model for Weather Prediction
Combines ARIMA and LSTM predictions
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from .arima_model import ARIMAPredictor
from .lstm_model import LSTMPredictor


class EnsemblePredictor:
    """
    Ensemble predictor that combines ARIMA and LSTM models
    Uses weighted averaging for final predictions
    """
    
    def __init__(self, arima_weight: float = 0.6, lstm_weight: float = 0.4):
        """
        Initialize ensemble predictor
        OPTIMIZED: ARIMA weighted slightly higher (60/40) as it's more stable for weather
        
        Args:
            arima_weight: Weight for ARIMA predictions (0-1), default 0.6
            lstm_weight: Weight for LSTM predictions (0-1), default 0.4
        """
        if abs(arima_weight + lstm_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.arima_weight = arima_weight
        self.lstm_weight = lstm_weight
        self.arima_predictor = ARIMAPredictor()
        self.lstm_predictor = None  # Initialize only if TensorFlow available
        self.models_trained = False
        self.predictions_cache = {}
    
    def train(self, temperature_data: pd.Series, humidity_data: pd.Series,
              lstm_epochs: int = 50, lstm_lookback: int = 24) -> Dict:
        """
        Train both ARIMA and LSTM models
        
        Args:
            temperature_data: Historical temperature readings
            humidity_data: Historical humidity readings
            lstm_epochs: Number of epochs for LSTM training
            lstm_lookback: Lookback window for LSTM
            
        Returns:
            Dictionary with training metrics for both models
        """
        training_info = {
            'ensemble_config': {
                'arima_weight': self.arima_weight,
                'lstm_weight': self.lstm_weight
            },
            'models': {}
        }
        
        # Train ARIMA model
        print("Training ARIMA model...")
        arima_info = self.arima_predictor.train(temperature_data, humidity_data)
        training_info['models']['arima'] = arima_info
        
        # Train LSTM model (if TensorFlow available)
        try:
            from .lstm_model import TENSORFLOW_AVAILABLE
            if TENSORFLOW_AVAILABLE:
                print("Training LSTM model...")
                self.lstm_predictor = LSTMPredictor(lookback=lstm_lookback)
                lstm_info = self.lstm_predictor.train(
                    temperature_data, 
                    humidity_data,
                    epochs=lstm_epochs,
                    verbose=0
                )
                training_info['models']['lstm'] = lstm_info
                training_info['lstm_available'] = True
            else:
                training_info['lstm_available'] = False
                training_info['models']['lstm'] = {
                    'success': False,
                    'error': 'TensorFlow not installed'
                }
        except Exception as e:
            training_info['lstm_available'] = False
            training_info['models']['lstm'] = {
                'success': False,
                'error': str(e)
            }
        
        self.models_trained = True
        return training_info
    
    def predict(self, temperature_data: pd.Series = None, humidity_data: pd.Series = None,
                steps: int = 5) -> Dict:
        """
        Generate ensemble predictions combining ARIMA and LSTM
        
        Args:
            temperature_data: Recent temperature data (needed for LSTM context)
            humidity_data: Recent humidity data (needed for LSTM context)
            steps: Number of hours to predict
            
        Returns:
            Dictionary with predictions from all models and ensemble
        """
        if not self.models_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {
            'steps': steps,
            'models': {}
        }
        
        # Get ARIMA predictions
        arima_temp, arima_humidity, arima_confidence = self.arima_predictor.predict(steps)
        predictions['models']['arima'] = {
            'temperature': arima_temp.tolist(),
            'humidity': arima_humidity.tolist(),
            'confidence': arima_confidence
        }
        
        # Get LSTM predictions (if available)
        if self.lstm_predictor is not None and temperature_data is not None:
            try:
                lstm_temp, lstm_humidity, lstm_confidence = self.lstm_predictor.predict(
                    temperature_data,
                    humidity_data,
                    steps
                )
                predictions['models']['lstm'] = {
                    'temperature': lstm_temp.tolist(),
                    'humidity': lstm_humidity.tolist(),
                    'confidence': lstm_confidence
                }
                
                # Calculate ensemble predictions (weighted average)
                ensemble_temp = (
                    self.arima_weight * arima_temp + 
                    self.lstm_weight * lstm_temp
                )
                ensemble_humidity = (
                    self.arima_weight * arima_humidity + 
                    self.lstm_weight * lstm_humidity
                )
                
                # Calculate ensemble confidence intervals
                ensemble_confidence = {
                    'temp_confidence_lower': [
                        self.arima_weight * a + self.lstm_weight * l
                        for a, l in zip(arima_confidence['temp_confidence_lower'],
                                      lstm_confidence['temp_confidence_lower'])
                    ],
                    'temp_confidence_upper': [
                        self.arima_weight * a + self.lstm_weight * l
                        for a, l in zip(arima_confidence['temp_confidence_upper'],
                                      lstm_confidence['temp_confidence_upper'])
                    ],
                    'humidity_confidence_lower': [
                        self.arima_weight * a + self.lstm_weight * l
                        for a, l in zip(arima_confidence['humidity_confidence_lower'],
                                      lstm_confidence['humidity_confidence_lower'])
                    ],
                    'humidity_confidence_upper': [
                        self.arima_weight * a + self.lstm_weight * l
                        for a, l in zip(arima_confidence['humidity_confidence_upper'],
                                      lstm_confidence['humidity_confidence_upper'])
                    ],
                    'confidence_level': 0.95
                }
                
                predictions['models']['ensemble'] = {
                    'temperature': ensemble_temp.tolist(),
                    'humidity': ensemble_humidity.tolist(),
                    'confidence': ensemble_confidence,
                    'method': 'weighted_average',
                    'weights': {
                        'arima': self.arima_weight,
                        'lstm': self.lstm_weight
                    }
                }
                
            except Exception as e:
                predictions['models']['lstm'] = {
                    'error': str(e),
                    'success': False
                }
                # Fallback to ARIMA only
                predictions['models']['ensemble'] = predictions['models']['arima'].copy()
                predictions['models']['ensemble']['method'] = 'arima_only'
        else:
            # No LSTM available, use ARIMA as ensemble
            predictions['models']['ensemble'] = predictions['models']['arima'].copy()
            predictions['models']['ensemble']['method'] = 'arima_only'
        
        self.predictions_cache = predictions
        return predictions
    
    def get_comparison_metrics(self) -> Dict:
        """
        Compare performance metrics between models
        
        Returns:
            Dictionary with comparative metrics
        """
        if not self.models_trained:
            return {'error': 'Models not trained yet'}
        
        comparison = {
            'arima': self.arima_predictor.get_model_info(),
            'ensemble_config': {
                'arima_weight': self.arima_weight,
                'lstm_weight': self.lstm_weight
            }
        }
        
        if self.lstm_predictor:
            comparison['lstm'] = self.lstm_predictor.get_model_info()
        
        return comparison
    
    def calculate_model_errors(self, actual_temp: List[float], actual_humidity: List[float]) -> Dict:
        """
        Calculate prediction errors for model evaluation
        
        Args:
            actual_temp: Actual temperature values
            actual_humidity: Actual humidity values
            
        Returns:
            Dictionary with MAE and RMSE for each model
        """
        if not self.predictions_cache:
            return {'error': 'No predictions available'}
        
        errors = {}
        
        for model_name, model_data in self.predictions_cache['models'].items():
            if 'temperature' in model_data and 'error' not in model_data:
                pred_temp = np.array(model_data['temperature'][:len(actual_temp)])
                pred_humidity = np.array(model_data['humidity'][:len(actual_humidity)])
                
                actual_temp_arr = np.array(actual_temp)
                actual_humidity_arr = np.array(actual_humidity)
                
                errors[model_name] = {
                    'temperature': {
                        'mae': float(np.mean(np.abs(pred_temp - actual_temp_arr))),
                        'rmse': float(np.sqrt(np.mean((pred_temp - actual_temp_arr)**2)))
                    },
                    'humidity': {
                        'mae': float(np.mean(np.abs(pred_humidity - actual_humidity_arr))),
                        'rmse': float(np.sqrt(np.mean((pred_humidity - actual_humidity_arr)**2)))
                    }
                }
        
        return errors
