#!/usr/bin/env python3
"""
Simple Demo of Demand Response Forecasting System
Shows core functionality without the full workflow complexity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_generator import DemandDataGenerator
from dr_forecasting_model import DemandResponseForecaster
from visualization import DRVisualization

def run_demo():
    print("="*60)
    print("DEMAND RESPONSE FORECASTING SYSTEM - DEMO")
    print("="*60)
    
    # Step 1: Generate sample historical data (smaller dataset for demo)
    print("\n1. Generating Historical Data...")
    print("-" * 30)
    
    generator = DemandDataGenerator(start_date='2023-12-01', end_date='2024-01-01')
    historical_data = generator.generate_historical_data()
    
    print(f"✓ Generated {len(historical_data)} hours of historical data")
    print(f"✓ Date range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"✓ Average demand: {historical_data['demand_mw'].mean():.1f} MW")
    
    # Step 2: Train the forecasting model
    print("\n2. Training Forecasting Model...")
    print("-" * 30)
    
    forecaster = DemandResponseForecaster()
    
    try:
        metrics = forecaster.train_models(historical_data)
        print(f"✓ Model trained successfully!")
        print(f"✓ Classification Accuracy: {metrics['classification_accuracy']:.3f}")
        print(f"✓ Demand Regression R²: {metrics['regression_r2']:.3f}")
        print(f"✓ Demand Regression MAE: {metrics['regression_mae']:.2f} MW")
    except Exception as e:
        print(f"✗ Training failed: {str(e)}")
        return
    
    # Step 3: Create forecast data
    print("\n3. Creating Forecast Data...")
    print("-" * 30)
    
    # Create simple forecast input data
    forecast_dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='h')[:-1]
    forecast_data = []
    
    for timestamp in forecast_dates:
        forecast_data.append({
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'month': timestamp.month,
            'is_weekend': timestamp.dayofweek >= 5,
            'is_holiday': False,
            'temperature': np.random.normal(75, 10),
            'humidity': np.random.uniform(40, 80)
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    print(f"✓ Created forecast data for {len(forecast_df)} hours")
    
    # Step 4: Run forecast
    print("\n4. Running DR Forecast...")
    print("-" * 30)
    
    try:
        results = forecaster.forecast_dr_events(forecast_df, dr_reduction_percent=15)
        
        # Filter for hours 8-17 only
        dr_hours = results[(results['hour'] >= 8) & (results['hour'] <= 17)]
        
        print(f"✓ Forecast completed!")
        print(f"✓ Total forecast hours (8-17): {len(dr_hours)}")
        print(f"✓ DR events recommended: {dr_hours['dr_recommended'].sum()}")
        print(f"✓ Total potential savings: {dr_hours['dr_savings_mw'].sum():.1f} MW")
        
        if dr_hours['dr_recommended'].sum() > 0:
            avg_savings = dr_hours[dr_hours['dr_recommended'] == 1]['dr_savings_mw'].mean()
            print(f"✓ Average savings per DR event: {avg_savings:.1f} MW")
        
    except Exception as e:
        print(f"✗ Forecast failed: {str(e)}")
        return
    
    # Step 5: Show sample results
    print("\n5. Sample Results...")
    print("-" * 30)
    
    # Show first few DR recommendations
    dr_events = dr_hours[dr_hours['dr_recommended'] == 1].head(5)
    
    if len(dr_events) > 0:
        print("Top DR Opportunities:")
        for _, row in dr_events.iterrows():
            print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%00')} - "
                  f"Baseline: {row['baseline_demand_mw']:.1f} MW, "
                  f"Savings: {row['dr_savings_mw']:.1f} MW, "
                  f"Probability: {row['dr_probability']:.2f}")
    else:
        print("No DR opportunities found in forecast period")
    
    # Step 6: Generate summary report
    print("\n6. Summary Report...")
    print("-" * 30)
    
    try:
        viz = DRVisualization()
        summary = viz.generate_dr_summary_report(dr_hours)
        print(summary)
    except Exception as e:
        print(f"Summary generation failed: {str(e)}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext Steps:")
    print("- Run 'python3 main_interface.py' for full workflow")
    print("- Check generated CSV files for detailed results")
    print("- View interactive dashboards in HTML files")
    print("- Customize parameters for your specific use case")

if __name__ == "__main__":
    run_demo()