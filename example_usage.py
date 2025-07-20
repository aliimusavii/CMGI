"""
Comprehensive Example: Short Term Load Forecasting with Neural Networks
This script demonstrates how to use the load forecasting system for real-world applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from load_forecasting_system import LoadForecastingSystem
from data_loader import LoadDataLoader
from advanced_models import AdvancedLoadForecasting

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_basic_forecasting():
    """
    Run basic load forecasting with multiple neural network models
    """
    print("=" * 60)
    print("BASIC LOAD FORECASTING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the forecasting system
    forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
    
    # Generate synthetic data
    print("\n1. Generating synthetic load data...")
    data = forecaster.generate_synthetic_data(n_samples=8760)  # 1 year of hourly data
    print(f"   Generated {len(data)} hourly samples")
    print(f"   Load range: {data['load'].min():.2f} - {data['load'].max():.2f} MW")
    
    # Display basic statistics
    print("\n2. Data Statistics:")
    print(data['load'].describe())
    
    # Prepare sequences for modeling
    print("\n3. Preparing sequences for training...")
    X, y = forecaster.prepare_sequences(data, features=['load'])
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Split data (temporal split to maintain time series structure)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Build and train different models
    models_to_train = {
        'LSTM': lambda: forecaster.build_lstm_model(X_train.shape[1:], units=[64, 32]),
        'GRU': lambda: forecaster.build_gru_model(X_train.shape[1:], units=[64, 32]),
        'CNN-LSTM': lambda: forecaster.build_cnn_lstm_model(X_train.shape[1:]),
        'Transformer': lambda: forecaster.build_transformer_model(X_train.shape[1:]),
        'Ensemble': lambda: forecaster.build_ensemble_model(X_train.shape[1:])
    }
    
    print("\n4. Training Models...")
    results = {}
    
    for name, model_builder in models_to_train.items():
        print(f"\n   Training {name} model...")
        model = model_builder()
        
        # Train the model
        history = forecaster.train_model(
            model, X_train, y_train, X_val, y_val, 
            name, epochs=30, batch_size=32
        )
        
        # Evaluate the model
        metrics = forecaster.evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        
        print(f"   {name} Results:")
        print(f"     RMSE: {metrics['RMSE']:.4f}")
        print(f"     MAE: {metrics['MAE']:.4f}")
        print(f"     MAPE: {metrics['MAPE']:.2f}%")
        print(f"     R²: {metrics['R²']:.4f}")
    
    # Compare all models
    print("\n5. Model Comparison:")
    comparison_df = forecaster.compare_models()
    
    # Plot results
    print("\n6. Generating visualizations...")
    forecaster.plot_training_history()
    forecaster.plot_predictions(n_samples=168)  # Show 1 week
    
    return forecaster, results

def run_advanced_forecasting():
    """
    Run advanced load forecasting with feature engineering and ensemble methods
    """
    print("\n" + "=" * 60)
    print("ADVANCED LOAD FORECASTING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize data loader and advanced forecaster
    loader = LoadDataLoader()
    advanced_forecaster = AdvancedLoadForecasting(sequence_length=24, prediction_horizon=1)
    
    # Generate comprehensive synthetic data with multiple features
    print("\n1. Generating comprehensive synthetic data...")
    data = loader.generate_comprehensive_synthetic_data(n_samples=8760)
    print(f"   Generated {len(data)} samples with {len(data.columns)} features")
    print(f"   Features: {list(data.columns)}")
    
    # Prepare features with engineering
    print("\n2. Feature engineering...")
    processed_data = loader.prepare_features(
        data, 
        target_col='load',
        include_lags=True,
        include_rolling=True,
        include_time=True
    )
    print(f"   Processed data shape: {processed_data.shape}")
    print(f"   Total features: {len(loader.features)}")
    
    # Prepare sequences with multiple features
    forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
    
    # Use selected features for modeling
    selected_features = [
        'load', 'temperature', 'humidity', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'is_weekend', 'load_lag_1', 'load_lag_24',
        'load_rolling_mean_24', 'cooling_load', 'heating_load'
    ]
    
    # Filter features that exist in the data
    available_features = [f for f in selected_features if f in processed_data.columns]
    
    X, y = forecaster.prepare_sequences(
        processed_data, 
        features=available_features
    )
    
    print(f"   Using {len(available_features)} features for modeling")
    print(f"   Sequence shape: {X.shape}")
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Train ensemble models
    print("\n3. Training ensemble models...")
    ensemble_models = advanced_forecaster.train_ensemble_models(
        X_train, y_train, X_val, y_val
    )
    
    # Calculate optimal weights
    print("\n4. Calculating optimal ensemble weights...")
    optimal_weights = advanced_forecaster.calculate_ensemble_weights(
        ensemble_models, X_val, y_val
    )
    
    print("   Optimal weights:")
    for name, weight in optimal_weights.items():
        print(f"     {name}: {weight:.4f}")
    
    # Make ensemble predictions
    print("\n5. Making ensemble predictions...")
    ensemble_pred = advanced_forecaster.ensemble_predict(
        ensemble_models, X_test, optimal_weights
    )
    
    # Evaluate ensemble performance
    ensemble_rmse = np.sqrt(mean_squared_error(y_test.ravel(), ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test.ravel(), ensemble_pred)
    ensemble_r2 = r2_score(y_test.ravel(), ensemble_pred)
    
    print(f"\n6. Ensemble Performance:")
    print(f"   RMSE: {ensemble_rmse:.4f}")
    print(f"   MAE: {ensemble_mae:.4f}")
    print(f"   R²: {ensemble_r2:.4f}")
    
    return advanced_forecaster, ensemble_models, optimal_weights

def run_hyperparameter_optimization():
    """
    Demonstrate hyperparameter optimization
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
    data = forecaster.generate_synthetic_data(n_samples=2000)  # Smaller dataset for speed
    X, y = forecaster.prepare_sequences(data, features=['load'])
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize advanced forecaster
    advanced_forecaster = AdvancedLoadForecasting(sequence_length=24, prediction_horizon=1)
    
    # Optimize different model types
    model_types = ['lstm', 'gru', 'cnn_lstm']
    
    for model_type in model_types:
        print(f"\n1. Optimizing {model_type.upper()} hyperparameters...")
        best_params = advanced_forecaster.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, 
            model_type=model_type, n_trials=20  # Reduced for demonstration
        )
        
        print(f"   Best parameters for {model_type}:")
        for param, value in best_params.items():
            print(f"     {param}: {value}")
    
    return advanced_forecaster.best_params

def run_real_time_forecasting_simulation():
    """
    Simulate real-time load forecasting
    """
    print("\n" + "=" * 60)
    print("REAL-TIME FORECASTING SIMULATION")
    print("=" * 60)
    
    # Initialize system
    forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
    
    # Generate data
    data = forecaster.generate_synthetic_data(n_samples=8760)
    X, y = forecaster.prepare_sequences(data, features=['load'])
    
    # Train a simple model for demonstration
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train LSTM model
    model = forecaster.build_lstm_model(X_train.shape[1:], units=[50])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Simulate real-time predictions
    print("\n1. Simulating real-time predictions...")
    predictions = []
    actuals = []
    
    # Take last 100 samples for simulation
    simulation_data = X_test[-100:]
    simulation_targets = y_test[-100:]
    
    for i in range(len(simulation_data)):
        # Make prediction for next hour
        current_sequence = simulation_data[i:i+1]
        pred = model.predict(current_sequence, verbose=0)[0][0]
        actual = simulation_targets[i]
        
        predictions.append(pred)
        actuals.append(actual)
        
        if i % 24 == 0:  # Print daily summary
            day = i // 24 + 1
            recent_mae = mean_absolute_error(actuals[-24:], predictions[-24:]) if i >= 23 else 0
            print(f"   Day {day}: Recent 24h MAE = {recent_mae:.4f}")
    
    # Final performance
    final_mae = mean_absolute_error(actuals, predictions)
    final_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    print(f"\n2. Real-time Simulation Results:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Average MAE: {final_mae:.4f}")
    print(f"   Average RMSE: {final_rmse:.4f}")
    
    # Plot real-time results
    plt.figure(figsize=(15, 6))
    plt.plot(actuals, label='Actual Load', linewidth=2)
    plt.plot(predictions, label='Predicted Load', linewidth=2, alpha=0.8)
    plt.title('Real-time Load Forecasting Simulation')
    plt.xlabel('Hours')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return predictions, actuals

def main():
    """
    Run comprehensive load forecasting demonstration
    """
    print("SHORT TERM LOAD FORECASTING WITH NEURAL NETWORKS")
    print("=" * 80)
    print("This demonstration covers:")
    print("1. Basic neural network models (LSTM, GRU, CNN-LSTM, Transformer, Ensemble)")
    print("2. Advanced feature engineering and ensemble methods")
    print("3. Hyperparameter optimization with Optuna")
    print("4. Real-time forecasting simulation")
    print("=" * 80)
    
    try:
        # Run basic forecasting
        basic_forecaster, basic_results = run_basic_forecasting()
        
        # Run advanced forecasting
        advanced_forecaster, ensemble_models, optimal_weights = run_advanced_forecasting()
        
        # Run hyperparameter optimization
        best_params = run_hyperparameter_optimization()
        
        # Run real-time simulation
        predictions, actuals = run_real_time_forecasting_simulation()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nKey takeaways:")
        print("1. Multiple neural network architectures can be effective for load forecasting")
        print("2. Feature engineering significantly improves performance")
        print("3. Ensemble methods often provide the best results")
        print("4. Hyperparameter optimization can fine-tune model performance")
        print("5. The system can be adapted for real-time forecasting applications")
        
        print("\nNext steps:")
        print("- Integrate with real electricity load data")
        print("- Add weather API integration for real-time features")
        print("- Implement model versioning and A/B testing")
        print("- Deploy models for production use")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please ensure all dependencies are installed (see requirements.txt)")

if __name__ == "__main__":
    main()