"""
Quick Demo: Short Term Load Forecasting
A simple demonstration that can be run immediately to test the system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
plt.switch_backend('Agg')

def run_quick_demo():
    """
    Run a quick demonstration of the load forecasting system
    """
    print("=" * 60)
    print("QUICK DEMO: SHORT TERM LOAD FORECASTING")
    print("=" * 60)
    
    try:
        from load_forecasting_system import LoadForecastingSystem
        
        # Initialize the system
        print("1. Initializing forecasting system...")
        forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)
        
        # Generate synthetic data
        print("2. Generating synthetic electricity load data...")
        data = forecaster.generate_synthetic_data(n_samples=2000)  # Smaller dataset for quick demo
        print(f"   Generated {len(data)} hourly samples")
        print(f"   Load range: {data['load'].min():.2f} - {data['load'].max():.2f} MW")
        
        # Show data statistics
        print("\n3. Data Statistics:")
        print(data['load'].describe())
        
        # Prepare sequences
        print("\n4. Preparing sequences for training...")
        X, y = forecaster.prepare_sequences(data, features=['load'])
        print(f"   Input shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        # Split data (temporal split)
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"   Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]} samples")
        
        # Train a simple LSTM model
        print("\n5. Training LSTM model...")
        model = forecaster.build_lstm_model(X_train.shape[1:], units=[32])
        
        history = forecaster.train_model(
            model, X_train, y_train, X_val, y_val, 
            'LSTM_Demo', epochs=20, batch_size=32
        )
        
        # Evaluate model
        print("\n6. Evaluating model performance...")
        metrics = forecaster.evaluate_model(model, X_test, y_test, 'LSTM_Demo')
        
        print("   Results:")
        print(f"     RMSE: {metrics['RMSE']:.4f}")
        print(f"     MAE: {metrics['MAE']:.4f}")
        print(f"     MAPE: {metrics['MAPE']:.2f}%")
        print(f"     R²: {metrics['R²']:.4f}")
        
        # Simple prediction visualization
        print("\n7. Creating visualization...")
        
        # Get predictions for plotting
        y_pred = model.predict(X_test[:100], verbose=0)  # First 100 test samples
        
        # Create a simple plot
        plt.figure(figsize=(12, 6))
        
        # Inverse transform for plotting (simplified)
        actual_values = forecaster.predictions['LSTM_Demo']['actual'][:100]
        predicted_values = forecaster.predictions['LSTM_Demo']['predicted'][:100]
        
        plt.plot(actual_values, label='Actual Load', linewidth=2, color='blue')
        plt.plot(predicted_values, label='Predicted Load', linewidth=2, color='red', alpha=0.8)
        
        plt.title('Load Forecasting Results - Quick Demo')
        plt.xlabel('Hours')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
        print("   Visualization saved as 'quick_demo_results.png'")
        
        # Show training progress
        training_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure(figsize=(10, 6))
        plt.plot(training_loss, label='Training Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='red')
        plt.title('Training Progress')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        print("   Training progress saved as 'training_progress.png'")
        
        print("\n" + "=" * 60)
        print("QUICK DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Results:")
        print(f"- Model Type: LSTM")
        print(f"- Training Samples: {X_train.shape[0]}")
        print(f"- Test RMSE: {metrics['RMSE']:.4f}")
        print(f"- Test MAE: {metrics['MAE']:.4f}")
        print(f"- Model R²: {metrics['R²']:.4f}")
        
        print("\nFiles Generated:")
        print("- quick_demo_results.png (forecasting results)")
        print("- training_progress.png (training history)")
        
        print("\nNext Steps:")
        print("1. Run 'python example_usage.py' for full demonstration")
        print("2. Explore different model architectures")
        print("3. Try with real electricity load data")
        print("4. Experiment with feature engineering")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please check your installation and try again.")
        return False

def main():
    """
    Main function to run the quick demo
    """
    success = run_quick_demo()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("Check the generated PNG files for visualizations.")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()