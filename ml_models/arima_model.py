"""
ARIMA Model for Weather Prediction
Statistical time series forecasting
"""
import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Dict


class ARIMAPredictor:
    """
    AutoRegressive Integrated Moving Average (ARIMA) model
    for time series weather prediction
    """
    
    def __init__(self):
        self.temp_model = None
        self.humidity_model = None
        self.temp_model_fit = None
        self.humidity_model_fit = None
        self.temp_params = None
        self.humidity_params = None
        self.training_history = {}
        
    def train(self, temperature_data: pd.Series, humidity_data: pd.Series) -> Dict:
        """
        Train ARIMA models for temperature and humidity
        OPTIMIZED: Expanded search space and better parameter tuning
        
        Args:
            temperature_data: Historical temperature readings
            humidity_data: Historical humidity readings
            
        Returns:
            Dictionary with training metrics
        """
        warnings.filterwarnings("ignore")
        
        training_info = {
            'model_type': 'ARIMA',
            'data_points': len(temperature_data),
            'success': True
        }
        
        try:
            # OPTIMIZED: Auto-tune ARIMA parameters for temperature
            # Expanded search space for better fitting
            temp_fit = auto_arima(
                temperature_data,
                start_p=1, start_q=1,
                max_p=5,              # Increased from 3 to 5
                max_q=5,              # Increased from 3 to 5
                max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                information_criterion='aic',  # Explicit AIC criterion
                n_fits=50,            # More model combinations tested
                stationary=False,     # Allow non-stationary data
                test='adf',           # Augmented Dickey-Fuller test
                maxiter=100           # More iterations for convergence
            )
            self.temp_params = temp_fit.get_params().get("order")
            training_info['temp_order'] = self.temp_params
            
            # Train temperature model with optimized parameters
            self.temp_model = ARIMA(
                temperature_data, 
                order=self.temp_params,
                enforce_stationarity=False,    # More flexible
                enforce_invertibility=False    # Better convergence
            )
            self.temp_model_fit = self.temp_model.fit()
            
            # Calculate training metrics
            temp_residuals = self.temp_model_fit.resid
            training_info['temp_mae'] = float(np.mean(np.abs(temp_residuals)))
            training_info['temp_rmse'] = float(np.sqrt(np.mean(temp_residuals**2)))
            training_info['temp_aic'] = float(self.temp_model_fit.aic)
            training_info['temp_bic'] = float(self.temp_model_fit.bic)
            
            # OPTIMIZED: Auto-tune ARIMA parameters for humidity
            humidity_fit = auto_arima(
                humidity_data,
                start_p=1, start_q=1,
                max_p=5,              # Increased from 3 to 5
                max_q=5,              # Increased from 3 to 5
                max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                information_criterion='aic',
                n_fits=50,
                stationary=False,
                test='adf',
                maxiter=100
            )
            self.humidity_params = humidity_fit.get_params().get("order")
            training_info['humidity_order'] = self.humidity_params
            
            # Train humidity model with optimized parameters
            self.humidity_model = ARIMA(
                humidity_data, 
                order=self.humidity_params,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.humidity_model_fit = self.humidity_model.fit()
            
            # Calculate training metrics
            humidity_residuals = self.humidity_model_fit.resid
            training_info['humidity_mae'] = float(np.mean(np.abs(humidity_residuals)))
            training_info['humidity_rmse'] = float(np.sqrt(np.mean(humidity_residuals**2)))
            training_info['humidity_aic'] = float(self.humidity_model_fit.aic)
            training_info['humidity_bic'] = float(self.humidity_model_fit.bic)
            
            self.training_history = training_info
            print(f"✓ ARIMA trained successfully")
            print(f"  Temperature order: {self.temp_params}, MAE: {training_info['temp_mae']:.2f}°C")
            print(f"  Humidity order: {self.humidity_params}, MAE: {training_info['humidity_mae']:.2f}%")
            return training_info
            
        except Exception as e:
            print(f"✗ ARIMA training failed: {str(e)}")
            training_info['success'] = False
            training_info['error'] = str(e)
            return training_info
    
    def predict(self, steps: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate predictions for future time steps
        
        Args:
            steps: Number of hours to predict (default: 5)
            
        Returns:
            Tuple of (temperature_predictions, humidity_predictions, confidence_info)
        """
        if self.temp_model_fit is None or self.humidity_model_fit is None:
            raise ValueError("Models must be trained before prediction")
        
        # Get predictions
        temp_forecast = self.temp_model_fit.forecast(steps=steps)
        humidity_forecast = self.humidity_model_fit.forecast(steps=steps)
        
        # Get confidence intervals (95%)
        temp_forecast_ci = self.temp_model_fit.get_forecast(steps=steps).conf_int(alpha=0.05)
        humidity_forecast_ci = self.humidity_model_fit.get_forecast(steps=steps).conf_int(alpha=0.05)
        
        confidence_info = {
            'temp_confidence_lower': temp_forecast_ci.iloc[:, 0].tolist(),
            'temp_confidence_upper': temp_forecast_ci.iloc[:, 1].tolist(),
            'humidity_confidence_lower': humidity_forecast_ci.iloc[:, 0].tolist(),
            'humidity_confidence_upper': humidity_forecast_ci.iloc[:, 1].tolist(),
            'confidence_level': 0.95
        }
        
        return temp_forecast.values, humidity_forecast.values, confidence_info
    
    def get_model_info(self) -> Dict:
        """Get model information and parameters"""
        return {
            'model_type': 'ARIMA',
            'temp_params': self.temp_params,
            'humidity_params': self.humidity_params,
            'training_history': self.training_history,
            'description': 'AutoRegressive Integrated Moving Average - Statistical time series model'
        }
