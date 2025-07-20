"""
Short Term Load Forecasting System using Neural Networks and Deep Learning
This module provides a comprehensive framework for electrical load forecasting
using various deep learning architectures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, 
    Flatten, Input, Concatenate, Attention, MultiHeadAttention,
    LayerNormalization, TimeDistributed, RepeatVector
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class LoadForecastingSystem:
    """
    A comprehensive load forecasting system using various deep learning models
    """
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        """
        Initialize the forecasting system
        
        Args:
            sequence_length (int): Number of historical hours to use for prediction
            prediction_horizon (int): Number of hours to forecast ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.history = {}
        self.predictions = {}
        
    def generate_synthetic_data(self, n_samples=8760, start_date='2023-01-01'):
        """
        Generate synthetic electrical load data with realistic patterns
        
        Args:
            n_samples (int): Number of hourly samples (8760 = 1 year)
            start_date (str): Start date for the time series
            
        Returns:
            pd.DataFrame: Synthetic load data with timestamps
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
    
    def prepare_sequences(self, data, features=['load'], target='load'):
        """
        Prepare sequences for time series prediction
        
        Args:
            data (pd.DataFrame): Input data
            features (list): List of feature columns to use
            target (str): Target column name
            
        Returns:
            tuple: (X, y) sequences ready for training
        """
        # Scale the data
        feature_data = data[features].values
        target_data = data[target].values.reshape(-1, 1)
        
        scaled_features = self.scaler.fit_transform(np.column_stack([feature_data, target_data]))
        scaled_target = scaled_features[:, -1]
        scaled_features = scaled_features[:, :-1] if len(features) > 1 else scaled_features[:, :-1]
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_target) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            if len(features) == 1:
                X.append(scaled_target[i:(i + self.sequence_length)])
            else:
                X.append(scaled_features[i:(i + self.sequence_length)])
            
            # Target (next prediction_horizon values)
            if self.prediction_horizon == 1:
                y.append(scaled_target[i + self.sequence_length])
            else:
                y.append(scaled_target[i + self.sequence_length:(i + self.sequence_length + self.prediction_horizon)])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for single feature
        if len(features) == 1:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def build_lstm_model(self, input_shape, units=[50, 50], dropout=0.2):
        """
        Build LSTM model for load forecasting
        
        Args:
            input_shape (tuple): Shape of input data
            units (list): Number of LSTM units in each layer
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units[0], return_sequences=True if len(units) > 1 else False),
            Dropout(dropout),
        ])
        
        # Add additional LSTM layers
        for i, unit in enumerate(units[1:]):
            model.add(LSTM(unit, return_sequences=True if i < len(units) - 2 else False))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_gru_model(self, input_shape, units=[50, 50], dropout=0.2):
        """
        Build GRU model for load forecasting
        
        Args:
            input_shape (tuple): Shape of input data
            units (list): Number of GRU units in each layer
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: Compiled GRU model
        """
        model = Sequential([
            Input(shape=input_shape),
            GRU(units[0], return_sequences=True if len(units) > 1 else False),
            Dropout(dropout),
        ])
        
        # Add additional GRU layers
        for i, unit in enumerate(units[1:]):
            model.add(GRU(unit, return_sequences=True if i < len(units) - 2 else False))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_cnn_lstm_model(self, input_shape, cnn_filters=[64, 32], lstm_units=[50], dropout=0.2):
        """
        Build CNN-LSTM hybrid model
        
        Args:
            input_shape (tuple): Shape of input data
            cnn_filters (list): Number of CNN filters in each layer
            lstm_units (list): Number of LSTM units
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: Compiled CNN-LSTM model
        """
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=cnn_filters[0], kernel_size=3, activation='relu'),
            Dropout(dropout),
        ])
        
        # Add additional CNN layers
        for filters in cnn_filters[1:]:
            model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
            model.add(Dropout(dropout))
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            model.add(LSTM(units, return_sequences=True if i < len(lstm_units) - 1 else False))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_transformer_model(self, input_shape, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        """
        Build Transformer model for load forecasting
        
        Args:
            input_shape (tuple): Shape of input data
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: Compiled Transformer model
        """
        inputs = Input(shape=input_shape)
        
        # Input projection
        x = Dense(d_model)(inputs)
        
        # Transformer blocks
        for _ in range(num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model//num_heads,
                dropout=dropout
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward
            ffn_output = Dense(d_model * 2, activation='relu')(x)
            ffn_output = Dropout(dropout)(ffn_output)
            ffn_output = Dense(d_model)(ffn_output)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Output layer
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_ensemble_model(self, input_shape):
        """
        Build an ensemble model combining multiple architectures
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled ensemble model
        """
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        lstm_branch = LSTM(50, return_sequences=False)(inputs)
        lstm_branch = Dropout(0.2)(lstm_branch)
        lstm_output = Dense(32, activation='relu')(lstm_branch)
        
        # GRU branch
        gru_branch = GRU(50, return_sequences=False)(inputs)
        gru_branch = Dropout(0.2)(gru_branch)
        gru_output = Dense(32, activation='relu')(gru_branch)
        
        # CNN branch
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        cnn_branch = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        cnn_output = Dense(32, activation='relu')(cnn_branch)
        
        # Combine branches
        combined = Concatenate()([lstm_output, gru_output, cnn_output])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        # Output layer
        outputs = Dense(self.prediction_horizon, activation='linear')(combined)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name, epochs=100, batch_size=32):
        """
        Train a model with callbacks
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name (str): Name of the model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.callbacks.History: Training history
        """
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint(f'best_{model_name}_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_name] = model
        self.history[model_name] = history
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions and actual values
        if self.prediction_horizon == 1:
            y_test_actual = self.scaler.inverse_transform(
                np.column_stack([np.zeros((len(y_test), X_test.shape[2] if len(X_test.shape) > 2 else 0)), 
                               y_test.reshape(-1, 1)]))[:, -1]
            y_pred_actual = self.scaler.inverse_transform(
                np.column_stack([np.zeros((len(y_pred), X_test.shape[2] if len(X_test.shape) > 2 else 0)), 
                               y_pred.reshape(-1, 1)]))[:, -1]
        else:
            # For multi-step prediction, handle differently
            y_test_actual = y_test
            y_pred_actual = y_pred
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
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
    
    def plot_training_history(self, model_names=None):
        """
        Plot training history for models
        
        Args:
            model_names (list): List of model names to plot
        """
        if model_names is None:
            model_names = list(self.history.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison', fontsize=16)
        
        for name in model_names:
            if name in self.history:
                history = self.history[name].history
                
                # Loss
                axes[0, 0].plot(history['loss'], label=f'{name} - Train')
                axes[0, 0].plot(history['val_loss'], label=f'{name} - Val')
                
                # MAE
                axes[0, 1].plot(history['mae'], label=f'{name} - Train')
                axes[0, 1].plot(history['val_mae'], label=f'{name} - Val')
        
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Remove empty subplots
        axes[1, 0].remove()
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, model_names=None, n_samples=200):
        """
        Plot predictions vs actual values
        
        Args:
            model_names (list): List of model names to plot
            n_samples (int): Number of samples to plot
        """
        if model_names is None:
            model_names = list(self.predictions.keys())
        
        fig = make_subplots(
            rows=len(model_names), cols=1,
            subplot_titles=[f'{name} Predictions' for name in model_names],
            vertical_spacing=0.05
        )
        
        for i, name in enumerate(model_names):
            if name in self.predictions:
                pred_data = self.predictions[name]
                actual = pred_data['actual'][:n_samples]
                predicted = pred_data['predicted'][:n_samples]
                
                # Actual values
                fig.add_trace(
                    go.Scatter(
                        y=actual,
                        mode='lines',
                        name=f'{name} - Actual',
                        line=dict(color='blue', width=1)
                    ),
                    row=i+1, col=1
                )
                
                # Predicted values
                fig.add_trace(
                    go.Scatter(
                        y=predicted,
                        mode='lines',
                        name=f'{name} - Predicted',
                        line=dict(color='red', width=1, dash='dash')
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Load Forecasting Results Comparison',
            height=300 * len(model_names),
            showlegend=True
        )
        
        fig.show()
    
    def compare_models(self):
        """
        Compare all trained models
        
        Returns:
            pd.DataFrame: Comparison table of model metrics
        """
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
    Main function to demonstrate the load forecasting system
    """
    print("Short Term Load Forecasting System")
    print("=" * 50)
    
    # Initialize the system
    forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
    
    # Generate synthetic data
    print("Generating synthetic load data...")
    data = forecaster.generate_synthetic_data(n_samples=8760)
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
        'LSTM': lambda: forecaster.build_lstm_model(X_train.shape[1:]),
        'GRU': lambda: forecaster.build_gru_model(X_train.shape[1:]),
        'CNN-LSTM': lambda: forecaster.build_cnn_lstm_model(X_train.shape[1:]),
        'Transformer': lambda: forecaster.build_transformer_model(X_train.shape[1:]),
        'Ensemble': lambda: forecaster.build_ensemble_model(X_train.shape[1:])
    }
    
    # Train models
    for name, model_builder in models_to_train.items():
        print(f"\nTraining {name} model...")
        model = model_builder()
        forecaster.train_model(
            model, X_train, y_train, X_val, y_val, 
            name, epochs=50, batch_size=32
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
    
    # Plot results
    print("\nGenerating visualizations...")
    forecaster.plot_training_history()
    forecaster.plot_predictions(n_samples=168)  # Show 1 week of predictions
    
    print("\nLoad forecasting analysis complete!")

if __name__ == "__main__":
    main()