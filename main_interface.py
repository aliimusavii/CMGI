"""
Main Interface for Demand Response Forecasting System
Provides command-line interface and workflow management for DR forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import sys

from data_generator import DemandDataGenerator
from dr_forecasting_model import DemandResponseForecaster
from visualization import DRVisualization

class DRForecastingInterface:
    def __init__(self):
        self.forecaster = DemandResponseForecaster()
        self.visualizer = DRVisualization()
        self.data_generator = DemandDataGenerator()
        
    def generate_sample_data(self, start_date='2023-01-01', end_date='2024-01-01', save_path=None):
        """Generate historical demand data for training"""
        print(f"Generating historical data from {start_date} to {end_date}...")
        
        generator = DemandDataGenerator(start_date, end_date)
        historical_data = generator.generate_historical_data()
        
        if save_path:
            historical_data.to_csv(save_path, index=False)
            print(f"Historical data saved to {save_path}")
        
        print(f"Generated {len(historical_data)} hours of data")
        print(f"Data summary:\n{historical_data.describe()}")
        
        return historical_data
    
    def train_model(self, data_path=None, historical_data=None, model_save_path=None):
        """Train the DR forecasting model"""
        if historical_data is None:
            if data_path and os.path.exists(data_path):
                print(f"Loading historical data from {data_path}...")
                historical_data = pd.read_csv(data_path)
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
            else:
                print("No data provided. Generating sample data...")
                historical_data = self.generate_sample_data()
        
        print("Training DR forecasting models...")
        metrics = self.forecaster.train_models(historical_data)
        
        if model_save_path:
            self.forecaster.save_model(model_save_path)
        
        print(f"Training completed successfully!")
        print(f"Model Performance Metrics:")
        print(f"- Classification Accuracy: {metrics['classification_accuracy']:.3f}")
        print(f"- Demand Regression R²: {metrics['regression_r2']:.3f}")
        print(f"- Demand Regression MAE: {metrics['regression_mae']:.2f} MW")
        
        return metrics
    
    def create_forecast_data(self, start_date, end_date, weather_data=None):
        """Create forecast input data with weather information"""
        print(f"Creating forecast data from {start_date} to {end_date}...")
        
        # Generate hourly timestamps
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')[:-1]
        
        forecast_data = []
        for timestamp in date_range:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            month = timestamp.month
            
            # Use provided weather data or generate synthetic weather
            if weather_data is not None:
                # Match timestamp with weather data
                weather_row = weather_data[weather_data['timestamp'] == timestamp]
                if not weather_row.empty:
                    temp = weather_row['temperature'].iloc[0]
                    humidity = weather_row['humidity'].iloc[0]
                else:
                    temp = self.data_generator.generate_temperature(month)
                    humidity = np.random.uniform(30, 90)
            else:
                temp = self.data_generator.generate_temperature(month)
                humidity = np.random.uniform(30, 90)
            
            forecast_data.append({
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': day_of_week >= 5,
                'is_holiday': False,  # Simplified - could integrate holidays
                'temperature': temp,
                'humidity': humidity
            })
        
        return pd.DataFrame(forecast_data)
    
    def run_forecast(self, start_date, end_date, model_path=None, 
                    dr_reduction_percent=15, weather_data=None):
        """Run DR forecast for specified period"""
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.forecaster.load_model(model_path)
        elif not self.forecaster.is_trained:
            print("No trained model found. Training new model...")
            self.train_model()
        
        # Create forecast input data
        forecast_data = self.create_forecast_data(start_date, end_date, weather_data)
        
        # Run forecast
        print(f"Running DR forecast for {len(forecast_data)} hours...")
        results = self.forecaster.forecast_dr_events(forecast_data, dr_reduction_percent)
        
        print(f"Forecast completed! Found {results['dr_recommended'].sum()} DR opportunities")
        
        return results
    
    def analyze_results(self, forecast_results, create_visualizations=True, save_plots=True):
        """Analyze and visualize forecast results"""
        print("Analyzing forecast results...")
        
        # Generate summary report
        summary = self.visualizer.generate_dr_summary_report(forecast_results)
        print(summary)
        
        if create_visualizations:
            print("Creating visualizations...")
            
            # Static plots
            if save_plots:
                self.visualizer.plot_dr_forecast_results(
                    forecast_results, 
                    save_path='/workspace/dr_forecast_results.png'
                )
            else:
                self.visualizer.plot_dr_forecast_results(forecast_results)
            
            # Interactive dashboard
            dashboard = self.visualizer.create_interactive_dr_dashboard(
                forecast_results,
                save_path='/workspace/dr_dashboard.html' if save_plots else None
            )
            
            if save_plots:
                print("Visualizations saved:")
                print("- Static plots: /workspace/dr_forecast_results.png")
                print("- Interactive dashboard: /workspace/dr_dashboard.html")
        
        return summary
    
    def run_complete_workflow(self, forecast_start, forecast_end, 
                            historical_start='2023-01-01', historical_end='2024-01-01',
                            dr_reduction_percent=15):
        """Run complete DR forecasting workflow"""
        print("=" * 60)
        print("DEMAND RESPONSE FORECASTING SYSTEM")
        print("=" * 60)
        
        # Step 1: Generate/Load historical data
        print("\n1. PREPARING HISTORICAL DATA")
        print("-" * 30)
        historical_data = self.generate_sample_data(
            historical_start, historical_end, 
            save_path='/workspace/historical_demand_data.csv'
        )
        
        # Step 2: Train model
        print("\n2. TRAINING FORECASTING MODEL")
        print("-" * 30)
        metrics = self.train_model(
            historical_data=historical_data,
            model_save_path='/workspace/dr_model.pkl'
        )
        
        # Step 3: Run forecast
        print("\n3. RUNNING DR FORECAST")
        print("-" * 30)
        forecast_results = self.run_forecast(
            forecast_start, forecast_end,
            dr_reduction_percent=dr_reduction_percent
        )
        
        # Step 4: Analyze results
        print("\n4. ANALYZING RESULTS")
        print("-" * 30)
        summary = self.analyze_results(forecast_results, create_visualizations=True)
        
        # Save results
        forecast_results.to_csv('/workspace/dr_forecast_results.csv', index=False)
        
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files created:")
        print("- historical_demand_data.csv: Training data")
        print("- dr_model.pkl: Trained forecasting model")
        print("- dr_forecast_results.csv: Forecast results")
        print("- dr_forecast_results.png: Static visualizations")
        print("- dr_dashboard.html: Interactive dashboard")
        
        return {
            'historical_data': historical_data,
            'forecast_results': forecast_results,
            'model_metrics': metrics,
            'summary': summary
        }

def main():
    parser = argparse.ArgumentParser(description='Demand Response Forecasting System')
    parser.add_argument('--mode', choices=['train', 'forecast', 'complete'], 
                       default='complete', help='Operation mode')
    parser.add_argument('--forecast-start', type=str, default='2024-01-01',
                       help='Forecast start date (YYYY-MM-DD)')
    parser.add_argument('--forecast-end', type=str, default='2024-01-07',
                       help='Forecast end date (YYYY-MM-DD)')
    parser.add_argument('--historical-start', type=str, default='2023-01-01',
                       help='Historical data start date (YYYY-MM-DD)')
    parser.add_argument('--historical-end', type=str, default='2024-01-01',
                       help='Historical data end date (YYYY-MM-DD)')
    parser.add_argument('--dr-reduction', type=float, default=15.0,
                       help='Expected DR reduction percentage')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to historical data CSV file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create interface
    interface = DRForecastingInterface()
    
    if args.mode == 'complete':
        # Run complete workflow
        results = interface.run_complete_workflow(
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
            historical_start=args.historical_start,
            historical_end=args.historical_end,
            dr_reduction_percent=args.dr_reduction
        )
        
    elif args.mode == 'train':
        # Train model only
        interface.train_model(
            data_path=args.data_path,
            model_save_path='/workspace/dr_model.pkl'
        )
        
    elif args.mode == 'forecast':
        # Run forecast only
        forecast_results = interface.run_forecast(
            start_date=args.forecast_start,
            end_date=args.forecast_end,
            model_path=args.model_path,
            dr_reduction_percent=args.dr_reduction
        )
        
        if not args.no_viz:
            interface.analyze_results(forecast_results)

if __name__ == "__main__":
    main()