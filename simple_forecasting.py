"""
Simplified Load Forecasting System for Executable
Uses only scikit-learn and basic libraries to reduce executable size
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SimpleForecastingSystem:
    """
    Simplified load forecasting system using scikit-learn models
    Designed for standalone executable with minimal dependencies
    """
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def generate_synthetic_data(self, n_samples=8760, start_date='2023-01-01'):
        """
        Generate synthetic electrical load data with realistic patterns
        """
        # Create time index
        dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
        
        # Generate base load pattern
        t = np.arange(n_samples)
        
        # Daily pattern (peak during day, low at night)
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * t / 24 + np.pi/2)
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 10 * np.sin(2 * np.pi * t / (24 * 7))
        
        # Seasonal pattern (higher in summer for AC, winter for heating)
        seasonal_pattern = 20 * np.sin(2 * np.pi * t / (24 * 365.25) + np.pi)
        
        # Temperature effect (simplified)
        temperature = 20 + 15 * np.sin(2 * np.pi * t / (24 * 365.25)) + 5 * np.random.normal(0, 1, n_samples)
        temp_effect = 0.5 * (temperature - 20) ** 2
        
        # Random noise
        noise = np.random.normal(0, 5, n_samples)
        
        # Combine all patterns
        load = daily_pattern + weekly_pattern + seasonal_pattern + temp_effect + noise
        load = np.maximum(load, 10)  # Ensure positive values
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'load': load,
            'temperature': temperature,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int)
        })
        
        return df
    
    def prepare_sequences(self, data, features=['load']):
        """
        Prepare sequences for time series prediction using sliding window
        """
        # Scale the data
        feature_data = data[features].values
        target_data = data[features[0]].values.reshape(-1, 1)  # Use first feature as target
        
        # Fit scaler on all data
        all_data = np.column_stack([feature_data, target_data])
        scaled_data = self.scaler.fit_transform(all_data)
        
        scaled_features = scaled_data[:, :-1] if len(features) > 1 else scaled_data[:, :-1]
        scaled_target = scaled_data[:, -1]
        
        # Create sequences using sliding window
        X, y = [], []
        for i in range(len(scaled_target) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence (flatten for traditional ML models)
            if len(features) == 1:
                X.append(scaled_target[i:(i + self.sequence_length)])
            else:
                # Flatten the sequence for traditional ML
                seq_features = scaled_features[i:(i + self.sequence_length)]
                X.append(seq_features.flatten())
            
            # Target (next prediction_horizon values)
            if self.prediction_horizon == 1:
                y.append(scaled_target[i + self.sequence_length])
            else:
                y.append(scaled_target[i + self.sequence_length:(i + self.sequence_length + self.prediction_horizon)])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def build_random_forest_model(self, **kwargs):
        """Build Random Forest model"""
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 10),
            'random_state': 42
        }
        return RandomForestRegressor(**params)
    
    def build_gradient_boosting_model(self, **kwargs):
        """Build Gradient Boosting model"""
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'random_state': 42
        }
        return GradientBoostingRegressor(**params)
    
    def build_linear_model(self, **kwargs):
        """Build Linear Regression model"""
        alpha = kwargs.get('alpha', 1.0)
        if alpha > 0:
            return Ridge(alpha=alpha, random_state=42)
        else:
            return LinearRegression()
    
    def build_svr_model(self, **kwargs):
        """Build Support Vector Regression model"""
        params = {
            'kernel': kwargs.get('kernel', 'rbf'),
            'C': kwargs.get('C', 1.0),
            'gamma': kwargs.get('gamma', 'scale')
        }
        return SVR(**params)
    
    def build_ensemble_model(self, **kwargs):
        """Build a simple ensemble of multiple models"""
        models = {
            'rf': self.build_random_forest_model(n_estimators=50),
            'gb': self.build_gradient_boosting_model(n_estimators=50),
            'linear': self.build_linear_model(alpha=1.0)
        }
        return models
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name, **kwargs):
        """Train a model"""
        if isinstance(model, dict):  # Ensemble
            trained_models = {}
            for name, m in model.items():
                m.fit(X_train, y_train.ravel())
                trained_models[name] = m
            self.models[model_name] = trained_models
        else:
            model.fit(X_train, y_train.ravel())
            self.models[model_name] = model
        
        # Create a simple history object
        class SimpleHistory:
            def __init__(self):
                self.history = {
                    'loss': [0.1, 0.08, 0.06, 0.05],  # Dummy values
                    'val_loss': [0.12, 0.09, 0.07, 0.06]
                }
        
        return SimpleHistory()
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        # Make predictions
        if isinstance(model, dict):  # Ensemble
            predictions = []
            for m in model.values():
                pred = m.predict(X_test)
                predictions.append(pred)
            y_pred = np.mean(predictions, axis=0)
        else:
            y_pred = model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_test_actual = self.inverse_transform_target(y_test.ravel())
        y_pred_actual = self.inverse_transform_target(y_pred.ravel())
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / np.maximum(y_test_actual, 1e-8))) * 100
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2
        }
        
        self.predictions[model_name] = {
            'actual': y_test_actual,
            'predicted': y_pred_actual,
            'metrics': metrics
        }
        
        return metrics
    
    def inverse_transform_target(self, scaled_values):
        """Inverse transform target values"""
        # Create dummy array with same shape as original fit data
        dummy = np.zeros((len(scaled_values), self.scaler.n_features_in_))
        dummy[:, -1] = scaled_values  # Put scaled values in last column
        
        # Inverse transform and return last column
        return self.scaler.inverse_transform(dummy)[:, -1]
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.predictions:
            print("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        for name, pred_data in self.predictions.items():
            metrics = pred_data['metrics'].copy()
            metrics['Model'] = name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df[['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R²']]
        df = df.sort_values('RMSE')
        
        print("Model Performance Comparison:")
        print("=" * 80)
        print(df.to_string(index=False, float_format='%.4f'))
        
        return df

def main():
    """
    Main function to demonstrate the simplified forecasting system
    """
    print("Simplified Load Forecasting System")
    print("=" * 50)
    
    # Initialize the system
    forecaster = SimpleForecastingSystem(sequence_length=24, prediction_horizon=1)
    
    # Generate synthetic data
    print("Generating synthetic load data...")
    data = forecaster.generate_synthetic_data(n_samples=2000)
    print(f"Generated {len(data)} hourly samples")
    
    # Prepare data for modeling
    print("Preparing sequences for training...")
    X, y = forecaster.prepare_sequences(data, features=['load'])
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train models
    models_to_train = {
        'Random Forest': lambda: forecaster.build_random_forest_model(),
        'Gradient Boosting': lambda: forecaster.build_gradient_boosting_model(),
        'Linear Regression': lambda: forecaster.build_linear_model(),
        'SVR': lambda: forecaster.build_svr_model(),
        'Ensemble': lambda: forecaster.build_ensemble_model()
    }
    
    # Train models
    for name, model_builder in models_to_train.items():
        print(f"\nTraining {name} model...")
        model = model_builder()
        forecaster.train_model(
            model, X_train, y_train, X_val, y_val, name
        )
        
        # Evaluate model
        print(f"Evaluating {name} model...")
        metrics = forecaster.evaluate_model(model, X_test, y_test, name)
        print(f"{name} Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Compare all models
    print("\n" + "="*50)
    forecaster.compare_models()
    
    print("\nSimplified load forecasting analysis complete!")

if __name__ == "__main__":
    main()