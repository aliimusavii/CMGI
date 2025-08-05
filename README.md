# Demand Response Forecasting System

A comprehensive machine learning system for forecasting demand response (DR) events during peak hours (8-17) when historical data contains no DR events. This system uses historical demand patterns to predict when demand response would be most beneficial.

## 🎯 Project Overview

This system addresses the challenge of forecasting demand response opportunities when:
- Historical data contains no DR events
- You need to predict DR opportunities for hours 8-17 only
- You want to optimize energy consumption during peak hours
- You need to estimate potential savings from DR implementation

## 🚀 Key Features

- **Smart DR Detection**: Uses machine learning to identify optimal DR opportunities
- **Historical Pattern Analysis**: Learns from past demand without DR events
- **Time-Focused Forecasting**: Specifically targets hours 8-17 for DR events
- **Comprehensive Visualization**: Interactive dashboards and static plots
- **Flexible Architecture**: Easy to adapt for different utilities and regions
- **Performance Metrics**: Detailed model evaluation and forecasting accuracy

## 📋 System Components

### 1. Data Generator (`data_generator.py`)
- Generates realistic historical demand data
- Includes weather patterns, seasonal variations, and time-of-day effects
- Creates baseline demand without DR events for model training

### 2. Forecasting Model (`dr_forecasting_model.py`)
- **Classification Model**: Predicts when DR events should occur
- **Regression Model**: Estimates demand levels and potential savings
- **Feature Engineering**: Creates time-based, weather, and demand pattern features
- **DR Opportunity Detection**: Identifies high-value DR periods

### 3. Visualization Tools (`visualization.py`)
- Historical demand pattern analysis
- DR forecast results visualization
- Interactive Plotly dashboards
- Model performance metrics
- Summary reports and statistics

### 4. Main Interface (`main_interface.py`)
- Command-line interface for all operations
- Complete workflow automation
- Flexible configuration options
- Result analysis and reporting

## 🛠️ Installation

### Prerequisites
```bash
# Install system packages (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-pip python3-pandas python3-numpy python3-sklearn \
                    python3-matplotlib python3-seaborn python3-plotly

# Install additional packages
pip3 install holidays joblib --break-system-packages
```

### Alternative Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 📊 Usage

### Quick Start
```bash
# Run complete workflow with default settings
python3 main_interface.py

# Custom date range
python3 main_interface.py --forecast-start 2024-01-01 --forecast-end 2024-01-31
```

### Advanced Usage

#### Train Model Only
```bash
python3 main_interface.py --mode train --data-path historical_data.csv
```

#### Run Forecast Only
```bash
python3 main_interface.py --mode forecast --model-path dr_model.pkl \
                         --forecast-start 2024-01-01 --forecast-end 2024-01-07
```

#### Custom DR Reduction
```bash
python3 main_interface.py --dr-reduction 20.0  # 20% demand reduction during DR
```

### Python API Usage

```python
from main_interface import DRForecastingInterface

# Initialize system
interface = DRForecastingInterface()

# Run complete workflow
results = interface.run_complete_workflow(
    forecast_start='2024-01-01',
    forecast_end='2024-01-07',
    dr_reduction_percent=15
)

# Access results
forecast_data = results['forecast_results']
model_metrics = results['model_metrics']
summary_report = results['summary']
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--forecast-start` | Start date for forecast (YYYY-MM-DD) | 2024-01-01 |
| `--forecast-end` | End date for forecast (YYYY-MM-DD) | 2024-01-07 |
| `--historical-start` | Historical data start date | 2023-01-01 |
| `--historical-end` | Historical data end date | 2024-01-01 |
| `--dr-reduction` | Expected DR reduction percentage | 15.0 |
| `--model-path` | Path to saved model file | None |
| `--data-path` | Path to historical data CSV | None |
| `--no-viz` | Skip visualization generation | False |

## 📈 Model Architecture

### Feature Engineering
- **Time Features**: Hour, day of week, month (with cyclical encoding)
- **Lag Features**: Previous hour, same hour previous day/week
- **Rolling Statistics**: 24-hour moving averages, standard deviations
- **Weather Features**: Temperature, humidity, interactions
- **Peak Indicators**: Business hours, super-peak periods
- **Demand Percentiles**: Relative demand within day

### DR Opportunity Criteria
1. **Time Window**: Must be during hours 8-17
2. **High Demand**: Above 90th percentile of recent demand
3. **Above Average**: Exceeds daily mean by 20%
4. **Weather Conditions**: Hot weather increases DR value
5. **Business Days**: Weekdays prioritized for commercial DR

### Model Performance
- **Classification**: Random Forest for DR event prediction
- **Regression**: Gradient Boosting for demand forecasting
- **Validation**: Cross-validation and holdout testing
- **Metrics**: Accuracy, precision, recall, R², MAE

## 📊 Output Files

After running the system, you'll get:

| File | Description |
|------|-------------|
| `historical_demand_data.csv` | Generated training data |
| `dr_model.pkl` | Trained machine learning model |
| `dr_forecast_results.csv` | Detailed forecast results |
| `dr_forecast_results.png` | Static visualization plots |
| `dr_dashboard.html` | Interactive dashboard |

## 🎯 Understanding Results

### Forecast Output Columns
- `timestamp`: Date and time of forecast
- `hour`: Hour of day (8-17 only)
- `baseline_demand_mw`: Predicted demand without DR
- `dr_probability`: Probability of DR recommendation (0-1)
- `dr_recommended`: Binary DR recommendation (0/1)
- `dr_savings_mw`: Estimated demand reduction from DR
- `adjusted_demand_mw`: Demand after DR implementation
- `temperature`: Weather condition

### Key Metrics
- **DR Events**: Number of recommended DR periods
- **Total Savings**: Cumulative demand reduction potential
- **Peak Hour**: Hour with highest DR probability
- **Success Rate**: Model accuracy on validation data

## 🔍 Example Results

```
DEMAND RESPONSE FORECAST SUMMARY
================================

Forecast Period: 2024-01-01 to 2024-01-07
Total Hours Analyzed: 70
DR Events Recommended: 14 (20.0%)

SAVINGS ANALYSIS:
- Total Potential Savings: 1,250.5 MW
- Average Savings per Event: 89.3 MW
- Peak DR Hour: 14:00 (35.2% probability)
```

## 🚀 Advanced Features

### Custom Weather Data
```python
# Provide your own weather forecast
weather_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=168, freq='h'),
    'temperature': [75, 76, 78, ...],
    'humidity': [45, 47, 52, ...]
})

results = interface.run_forecast(
    start_date='2024-01-01',
    end_date='2024-01-07',
    weather_data=weather_data
)
```

### Model Customization
```python
# Adjust DR criteria
forecaster = DemandResponseForecaster()
# Modify thresholds in identify_dr_opportunities method
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed
2. **Date Format**: Use YYYY-MM-DD format for dates
3. **Memory Issues**: Reduce historical data range for large datasets
4. **Visualization**: Install plotly for interactive dashboards

### Performance Optimization

- Use smaller date ranges for faster processing
- Pre-filter data to hours 8-17 for efficiency
- Cache trained models for repeated forecasting
- Use parallel processing for large datasets

## 📚 Technical Details

### Algorithm Overview
1. **Data Preprocessing**: Clean and engineer features from historical data
2. **DR Labeling**: Identify periods where DR would have been beneficial
3. **Model Training**: Train classification and regression models
4. **Forecasting**: Predict DR opportunities for future periods
5. **Optimization**: Calculate optimal DR timing and savings

### Validation Strategy
- Time-series cross-validation
- Holdout testing on recent data
- Performance metrics across different seasons
- Robustness testing with various weather conditions

## 🤝 Contributing

This system is designed for educational and research purposes. Key areas for enhancement:

- Integration with real-time weather APIs
- Support for multiple utility territories
- Advanced DR optimization algorithms
- Real-time model updating capabilities

## 📄 License

MIT License - See LICENSE file for details.

## 🎓 Educational Context

This project was developed for AI/ML coursework, demonstrating:
- Time series forecasting techniques
- Feature engineering for energy data
- Machine learning model deployment
- Visualization and reporting systems
- Command-line interface design

---

**Note**: This system generates synthetic data for demonstration. In production, integrate with actual historical demand and weather data sources for optimal accuracy.
