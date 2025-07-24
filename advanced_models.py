"""
Advanced Neural Network Models for Load Forecasting
Includes hyperparameter optimization, ensemble methods, and advanced architectures
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

class AdvancedLoadForecasting:
    """
    Advanced load forecasting with hyperparameter optimization and ensemble methods
    """
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.best_params = {}
        self.models = {}
        
    def build_attention_lstm(self, input_shape, lstm_units=50, attention_units=32, dropout=0.2):
        """
        Build LSTM with attention mechanism
        
        Args:
            input_shape (tuple): Input shape
            lstm_units (int): Number of LSTM units
            attention_units (int): Number of attention units
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: LSTM model with attention
        """
        inputs = Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = LSTM(lstm_units, return_sequences=True, dropout=dropout)(inputs)
        
        # Attention mechanism
        attention = Dense(attention_units, activation='tanh')(lstm_out)
        attention = Dense(1, activation='softmax')(attention)
        
        # Apply attention weights
        context = Multiply()([lstm_out, attention])
        context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Output layer
        outputs = Dense(self.prediction_horizon, activation='linear')(context)
        
        model = Model(inputs, outputs)
        return model
    
    def build_wavenet_model(self, input_shape, filters=32, kernel_size=2, 
                           dilation_rates=[1, 2, 4, 8], dropout=0.2):
        """
        Build WaveNet-inspired model for load forecasting
        
        Args:
            input_shape (tuple): Input shape
            filters (int): Number of filters
            kernel_size (int): Kernel size
            dilation_rates (list): Dilation rates for dilated convolutions
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: WaveNet model
        """
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv1D(filters, kernel_size, padding='causal', activation='relu')(inputs)
        
        # Dilated convolution blocks
        skip_connections = []
        for dilation_rate in dilation_rates:
            # Dilated convolution
            conv_out = Conv1D(filters, kernel_size, 
                             dilation_rate=dilation_rate, 
                             padding='causal', activation='tanh')(x)
            
            # Gated activation
            gate_out = Conv1D(filters, kernel_size, 
                             dilation_rate=dilation_rate, 
                             padding='causal', activation='sigmoid')(x)
            
            gated = Multiply()([conv_out, gate_out])
            
            # Skip connection
            skip = Conv1D(filters, 1, activation='relu')(gated)
            skip_connections.append(skip)
            
            # Residual connection
            residual = Conv1D(filters, 1)(gated)
            x = Add()([x, residual])
        
        # Combine skip connections
        skip_sum = Add()(skip_connections)
        
        # Final layers
        x = Activation('relu')(skip_sum)
        x = Conv1D(filters, 1, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Conv1D(self.prediction_horizon, 1)(x)
        
        # Global average pooling
        outputs = GlobalAveragePooling1D()(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_resnet_model(self, input_shape, filters=64, n_blocks=3, dropout=0.2):
        """
        Build ResNet-inspired model for time series
        
        Args:
            input_shape (tuple): Input shape
            filters (int): Number of filters
            n_blocks (int): Number of residual blocks
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: ResNet model
        """
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv1D(filters, 7, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        # Residual blocks
        for i in range(n_blocks):
            # Main path
            shortcut = x
            
            x = Conv1D(filters, 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
            
            x = Conv1D(filters, 3, padding='same')(x)
            x = BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = Conv1D(filters, 1, padding='same')(shortcut)
            
            # Add shortcut
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            filters *= 2  # Increase filters in deeper blocks
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Final layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout)(x)
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_lstm_autoencoder(self, input_shape, encoding_dim=32, lstm_units=50, dropout=0.2):
        """
        Build LSTM Autoencoder for anomaly detection and forecasting
        
        Args:
            input_shape (tuple): Input shape
            encoding_dim (int): Encoding dimension
            lstm_units (int): Number of LSTM units
            dropout (float): Dropout rate
            
        Returns:
            keras.Model: LSTM Autoencoder model
        """
        inputs = Input(shape=input_shape)
        
        # Encoder
        encoded = LSTM(lstm_units, activation='relu', dropout=dropout)(inputs)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = RepeatVector(input_shape[0])(encoded)
        decoded = LSTM(lstm_units, return_sequences=True, activation='relu', dropout=dropout)(decoded)
        
        # Reconstruction output
        reconstruction = TimeDistributed(Dense(input_shape[1]))(decoded)
        
        # Forecasting branch
        forecast = Dense(32, activation='relu')(encoded)
        forecast = Dropout(dropout)(forecast)
        forecast_output = Dense(self.prediction_horizon, activation='linear')(forecast)
        
        model = Model(inputs, [reconstruction, forecast_output])
        return model
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, 
                                model_type='lstm', n_trials=50):
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type (str): Type of model to optimize
            n_trials (int): Number of optimization trials
            
        Returns:
            dict: Best hyperparameters
        """
        def objective(trial):
            # Common hyperparameters
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            if model_type == 'lstm':
                units1 = trial.suggest_int('lstm_units1', 32, 128)
                units2 = trial.suggest_int('lstm_units2', 16, 64)
                
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    LSTM(units1, return_sequences=True, dropout=dropout),
                    LSTM(units2, dropout=dropout),
                    Dense(self.prediction_horizon, activation='linear')
                ])
                
            elif model_type == 'gru':
                units1 = trial.suggest_int('gru_units1', 32, 128)
                units2 = trial.suggest_int('gru_units2', 16, 64)
                
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    GRU(units1, return_sequences=True, dropout=dropout),
                    GRU(units2, dropout=dropout),
                    Dense(self.prediction_horizon, activation='linear')
                ])
                
            elif model_type == 'cnn_lstm':
                filters1 = trial.suggest_int('cnn_filters1', 32, 128)
                filters2 = trial.suggest_int('cnn_filters2', 16, 64)
                lstm_units = trial.suggest_int('lstm_units', 32, 128)
                
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    Conv1D(filters1, 3, activation='relu'),
                    Dropout(dropout),
                    Conv1D(filters2, 3, activation='relu'),
                    Dropout(dropout),
                    LSTM(lstm_units, dropout=dropout),
                    Dense(self.prediction_horizon, activation='linear')
                ])
            
            elif model_type == 'attention_lstm':
                lstm_units = trial.suggest_int('lstm_units', 32, 128)
                attention_units = trial.suggest_int('attention_units', 16, 64)
                
                model = self.build_attention_lstm(
                    X_train.shape[1:], lstm_units, attention_units, dropout
                )
            
            # Compile model
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
            if optimizer_name == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            else:
                optimizer = RMSprop(learning_rate=learning_rate)
            
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Train model
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return validation loss
            return min(history.history['val_loss'])
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params[model_type] = study.best_params
        return study.best_params
    
    def build_stacking_ensemble(self, input_shape, base_models_configs):
        """
        Build stacking ensemble model
        
        Args:
            input_shape (tuple): Input shape
            base_models_configs (list): List of base model configurations
            
        Returns:
            keras.Model: Stacking ensemble model
        """
        inputs = Input(shape=input_shape)
        
        # Base models
        base_outputs = []
        for i, config in enumerate(base_models_configs):
            if config['type'] == 'lstm':
                base_model = LSTM(config['units'], dropout=config['dropout'], 
                                name=f'lstm_{i}')(inputs)
            elif config['type'] == 'gru':
                base_model = GRU(config['units'], dropout=config['dropout'], 
                               name=f'gru_{i}')(inputs)
            elif config['type'] == 'cnn':
                base_model = Conv1D(config['filters'], 3, activation='relu', 
                                  name=f'cnn_{i}')(inputs)
                base_model = GlobalAveragePooling1D()(base_model)
            
            base_output = Dense(32, activation='relu')(base_model)
            base_outputs.append(base_output)
        
        # Meta-learner
        combined = Concatenate()(base_outputs)
        meta_output = Dense(64, activation='relu')(combined)
        meta_output = Dropout(0.3)(meta_output)
        meta_output = Dense(32, activation='relu')(meta_output)
        outputs = Dense(self.prediction_horizon, activation='linear')(meta_output)
        
        model = Model(inputs, outputs)
        return model
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models for ensemble
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained ensemble models
        """
        ensemble_models = {}
        
        # Neural network models
        nn_configs = {
            'lstm': {'units': [64, 32], 'dropout': 0.2},
            'gru': {'units': [64, 32], 'dropout': 0.2},
            'cnn_lstm': {'filters': [64, 32], 'lstm_units': 50, 'dropout': 0.2},
            'attention_lstm': {'lstm_units': 64, 'attention_units': 32, 'dropout': 0.2}
        }
        
        for name, config in nn_configs.items():
            print(f"Training {name} model...")
            
            if name == 'lstm':
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    LSTM(config['units'][0], return_sequences=True, dropout=config['dropout']),
                    LSTM(config['units'][1], dropout=config['dropout']),
                    Dense(self.prediction_horizon, activation='linear')
                ])
            elif name == 'gru':
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    GRU(config['units'][0], return_sequences=True, dropout=config['dropout']),
                    GRU(config['units'][1], dropout=config['dropout']),
                    Dense(self.prediction_horizon, activation='linear')
                ])
            elif name == 'cnn_lstm':
                model = Sequential([
                    Input(shape=X_train.shape[1:]),
                    Conv1D(config['filters'][0], 3, activation='relu'),
                    Dropout(config['dropout']),
                    Conv1D(config['filters'][1], 3, activation='relu'),
                    Dropout(config['dropout']),
                    LSTM(config['lstm_units'], dropout=config['dropout']),
                    Dense(self.prediction_horizon, activation='linear')
                ])
            elif name == 'attention_lstm':
                model = self.build_attention_lstm(
                    X_train.shape[1:], 
                    config['lstm_units'], 
                    config['attention_units'], 
                    config['dropout']
                )
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
            ]
            
            model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            ensemble_models[name] = model
        
        # Traditional ML models (for comparison)
        # Reshape data for traditional ML
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_2d, y_train.ravel())
        ensemble_models['random_forest'] = rf_model
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_2d, y_train.ravel())
        ensemble_models['xgboost'] = xgb_model
        
        # LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(X_train_2d, y_train.ravel())
        ensemble_models['lightgbm'] = lgb_model
        
        return ensemble_models
    
    def ensemble_predict(self, models, X_test, weights=None):
        """
        Make ensemble predictions
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test data
            weights (dict): Weights for each model
            
        Returns:
            np.array: Ensemble predictions
        """
        predictions = {}
        
        # Get predictions from each model
        for name, model in models.items():
            if name in ['random_forest', 'xgboost', 'lightgbm']:
                # Traditional ML models
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                pred = model.predict(X_test_2d)
            else:
                # Neural network models
                pred = model.predict(X_test, verbose=0)
            
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions[name] = pred
        
        # Convert to DataFrame for easier handling
        pred_df = pd.DataFrame(predictions)
        
        if weights is None:
            # Equal weights
            ensemble_pred = pred_df.mean(axis=1).values
        else:
            # Weighted average
            weighted_preds = []
            for name, weight in weights.items():
                if name in pred_df.columns:
                    weighted_preds.append(weight * pred_df[name])
            ensemble_pred = sum(weighted_preds)
        
        return ensemble_pred
    
    def calculate_ensemble_weights(self, models, X_val, y_val):
        """
        Calculate optimal ensemble weights based on validation performance
        
        Args:
            models (dict): Dictionary of trained models
            X_val, y_val: Validation data
            
        Returns:
            dict: Optimal weights for each model
        """
        # Get validation predictions
        val_predictions = {}
        val_errors = {}
        
        for name, model in models.items():
            if name in ['random_forest', 'xgboost', 'lightgbm']:
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
                pred = model.predict(X_val_2d)
            else:
                pred = model.predict(X_val, verbose=0)
            
            if pred.ndim > 1:
                pred = pred.ravel()
            
            val_predictions[name] = pred
            val_errors[name] = mean_squared_error(y_val.ravel(), pred)
        
        # Calculate weights inversely proportional to error
        total_inverse_error = sum(1 / error for error in val_errors.values())
        weights = {name: (1 / error) / total_inverse_error 
                  for name, error in val_errors.items()}
        
        return weights
    
    def cross_validate_ensemble(self, X, y, n_splits=5):
        """
        Perform time series cross-validation for ensemble
        
        Args:
            X, y: Full dataset
            n_splits (int): Number of CV splits
            
        Returns:
            dict: Cross-validation scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'lstm': [], 'gru': [], 'cnn_lstm': [], 'attention_lstm': [],
            'random_forest': [], 'xgboost': [], 'lightgbm': [], 'ensemble': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train ensemble models
            models = self.train_ensemble_models(X_train, y_train, X_val, y_val)
            
            # Calculate weights
            weights = self.calculate_ensemble_weights(models, X_val, y_val)
            
            # Evaluate each model
            for name, model in models.items():
                if name in ['random_forest', 'xgboost', 'lightgbm']:
                    X_val_2d = X_val.reshape(X_val.shape[0], -1)
                    pred = model.predict(X_val_2d)
                else:
                    pred = model.predict(X_val, verbose=0)
                
                if pred.ndim > 1:
                    pred = pred.ravel()
                
                score = mean_squared_error(y_val.ravel(), pred)
                cv_scores[name].append(score)
            
            # Evaluate ensemble
            ensemble_pred = self.ensemble_predict(models, X_val, weights)
            ensemble_score = mean_squared_error(y_val.ravel(), ensemble_pred)
            cv_scores['ensemble'].append(ensemble_score)
        
        # Calculate mean scores
        mean_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}
        
        return mean_scores, cv_scores

def main():
    """
    Demonstration of advanced load forecasting techniques
    """
    print("Advanced Load Forecasting System")
    print("=" * 50)
    
    # This would typically use real data
    print("This module provides advanced neural network architectures")
    print("for load forecasting including:")
    print("- Attention-based LSTM")
    print("- WaveNet-inspired models")
    print("- ResNet for time series")
    print("- LSTM Autoencoders")
    print("- Hyperparameter optimization with Optuna")
    print("- Ensemble methods with stacking")
    print("- Cross-validation for time series")
    
    print("\nTo use these models, import this module and integrate")
    print("with your load forecasting pipeline.")

if __name__ == "__main__":
    main()