# Short Term Load Forecasting with Neural Networks and Deep Learning

A comprehensive system for electrical load forecasting using various deep learning architectures including LSTM, GRU, CNN-LSTM, Transformers, and ensemble methods.

## 🚀 Features

- **Multiple Neural Network Architectures**:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - CNN-LSTM Hybrid Models
  - Transformer Models
  - Attention-based LSTM
  - WaveNet-inspired Models
  - ResNet for Time Series
  - LSTM Autoencoders

- **Advanced Capabilities**:
  - Ensemble methods with stacking
  - Hyperparameter optimization using Optuna
  - Comprehensive feature engineering
  - Time series cross-validation
  - Real-time forecasting simulation
  - Interactive visualizations

- **Data Handling**:
  - Synthetic data generation with realistic patterns
  - Multiple external factors (weather, holidays, etc.)
  - Lag and rolling features
  - Cyclical time encoding

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- scikit-learn
- pandas
- numpy
- matplotlib
- plotly
- optuna
- xgboost
- lightgbm

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CMGI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Usage

```python
from load_forecasting_system import LoadForecastingSystem

# Initialize the system
forecaster = LoadForecastingSystem(sequence_length=24, prediction_horizon=1)

# Generate synthetic data
data = forecaster.generate_synthetic_data(n_samples=8760)

# Prepare sequences
X, y = forecaster.prepare_sequences(data, features=['load'])

# Build and train LSTM model
model = forecaster.build_lstm_model(X.shape[1:])
forecaster.train_model(model, X_train, y_train, X_val, y_val, 'LSTM')

# Evaluate model
metrics = forecaster.evaluate_model(model, X_test, y_test, 'LSTM')
```

### Run Complete Example

```bash
python example_usage.py
```

This will run a comprehensive demonstration including:
- Basic neural network models comparison
- Advanced feature engineering
- Ensemble methods
- Hyperparameter optimization
- Real-time forecasting simulation

## 📊 Model Architectures

### 1. LSTM (Long Short-Term Memory)
- Handles long-term dependencies in time series
- Configurable layers and units
- Dropout for regularization

### 2. GRU (Gated Recurrent Unit)
- Simplified LSTM variant
- Faster training with similar performance
- Good for shorter sequences

### 3. CNN-LSTM Hybrid
- Convolutional layers for pattern extraction
- LSTM layers for temporal modeling
- Effective for complex patterns

### 4. Transformer
- Self-attention mechanism
- Parallel processing
- State-of-the-art for sequence modeling

### 5. Ensemble Models
- Combines multiple architectures
- Weighted voting based on validation performance
- Typically provides best results

## 🔧 Advanced Features

### Hyperparameter Optimization

```python
from advanced_models import AdvancedLoadForecasting

forecaster = AdvancedLoadForecasting()
best_params = forecaster.optimize_hyperparameters(
    X_train, y_train, X_val, y_val, 
    model_type='lstm', n_trials=100
)
```

### Feature Engineering

```python
from data_loader import LoadDataLoader

loader = LoadDataLoader()
data = loader.generate_comprehensive_synthetic_data()
processed_data = loader.prepare_features(
    data, 
    include_lags=True,
    include_rolling=True,
    include_time=True
)
```

### Ensemble Training

```python
ensemble_models = forecaster.train_ensemble_models(
    X_train, y_train, X_val, y_val
)
weights = forecaster.calculate_ensemble_weights(
    ensemble_models, X_val, y_val
)
predictions = forecaster.ensemble_predict(
    ensemble_models, X_test, weights
)
```

## 📈 Performance Metrics

The system evaluates models using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## 🎨 Visualizations

The system provides interactive visualizations:
- Training history plots
- Prediction vs actual comparisons
- Real-time forecasting charts
- Feature importance analysis

## 📁 Project Structure

```
CMGI/
├── load_forecasting_system.py  # Main forecasting system
├── data_loader.py              # Data handling and feature engineering
├── advanced_models.py          # Advanced architectures and optimization
├── example_usage.py            # Comprehensive examples
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔮 Use Cases

- **Utility Companies**: Grid planning and optimization
- **Energy Trading**: Market forecasting and bidding
- **Industrial Facilities**: Demand planning and cost optimization
- **Smart Grids**: Real-time load balancing
- **Research**: Algorithm development and comparison

## 🎓 Educational Value

This project demonstrates:
- Time series forecasting concepts
- Deep learning for sequential data
- Feature engineering techniques
- Model evaluation and comparison
- Ensemble methods
- Hyperparameter optimization
- Real-world application development

## 🔧 Customization

### Adding New Models

```python
def build_custom_model(self, input_shape):
    inputs = Input(shape=input_shape)
    # Add your custom architecture here
    outputs = Dense(self.prediction_horizon)(x)
    model = Model(inputs, outputs)
    return model
```

### Custom Features

```python
def add_custom_features(self, data):
    # Add domain-specific features
    data['custom_feature'] = compute_custom_feature(data)
    return data
```

## 📚 References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Time Series Forecasting](https://otexts.com/fpp3/)
- [Ensemble Methods](https://link.springer.com/article/10.1023/A:1010933404324)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created for AI Class - Demonstrating practical applications of deep learning in energy forecasting.

---

**Note**: This system uses synthetic data for demonstration. For production use, integrate with real electricity load data and weather APIs.