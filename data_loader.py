"""
Data Loader Module for Load Forecasting
Handles real-world electricity load data, weather data, and external factors
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LoadDataLoader:
    """
    Data loader for electricity load forecasting with external factors
    """
    
    def __init__(self):
        self.data = None
        self.features = None
        
    def load_sample_data(self, file_path=None):
        """
        Load sample electricity load data
        
        Args:
            file_path (str): Path to CSV file with load data
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if file_path and os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            # Generate synthetic data if no file provided
            print("No data file provided. Generating synthetic data...")
            data = self.generate_comprehensive_synthetic_data()
        
        self.data = data
        return data
    
    def generate_comprehensive_synthetic_data(self, n_samples=8760, start_date='2023-01-01'):
        """
        Generate comprehensive synthetic electricity load data with multiple factors
        
        Args:
            n_samples (int): Number of hourly samples
            start_date (str): Start date for the time series
            
        Returns:
            pd.DataFrame: Synthetic load data with multiple features
        """
        # Create time index
        dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
        
        # Time-based features
        t = np.arange(n_samples)
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month
        day_of_year = dates.dayofyear
        
        # Base patterns
        # Daily pattern (peak during business hours)
        daily_pattern = 50 + 25 * np.sin(2 * np.pi * (hour - 6) / 24)
        daily_pattern += 10 * np.sin(4 * np.pi * (hour - 6) / 24)  # Sub-harmonics
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 15 * (1 - 0.3 * np.sin(2 * np.pi * day_of_week / 7))
        
        # Seasonal pattern
        seasonal_pattern = 20 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Weather simulation
        base_temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)
        temperature = base_temp + 5 * np.random.normal(0, 1, n_samples)
        
        # Humidity (inverse correlation with temperature)
        humidity = 70 - 0.5 * (temperature - 20) + 10 * np.random.normal(0, 1, n_samples)
        humidity = np.clip(humidity, 20, 95)
        
        # Wind speed
        wind_speed = 5 + 3 * np.sin(2 * np.pi * day_of_year / 365.25) + 2 * np.random.exponential(1, n_samples)
        wind_speed = np.clip(wind_speed, 0, 15)
        
        # Solar irradiance (simplified)
        solar_irradiance = np.maximum(0, 800 * np.sin(np.pi * (hour - 6) / 12) * 
                                    (1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)))
        solar_irradiance *= (1 + 0.2 * np.random.normal(0, 1, n_samples))
        solar_irradiance = np.maximum(solar_irradiance, 0)
        
        # Economic factors
        electricity_price = 0.12 + 0.03 * np.sin(2 * np.pi * day_of_year / 365.25) + \
                           0.01 * np.random.normal(0, 1, n_samples)
        electricity_price = np.maximum(electricity_price, 0.05)
        
        # Special events (holidays, etc.)
        holidays = self.generate_holiday_indicators(dates)
        
        # Temperature effect on load
        cooling_effect = np.maximum(0, (temperature - 22) * 2)  # AC usage
        heating_effect = np.maximum(0, (18 - temperature) * 1.5)  # Heating usage
        temp_effect = cooling_effect + heating_effect
        
        # Industrial load pattern
        industrial_base = 30 * (1 - 0.4 * holidays)  # Reduced during holidays
        industrial_pattern = industrial_base * (1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7))
        
        # Residential load pattern
        residential_base = 25
        residential_pattern = residential_base * (1.5 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 24))
        
        # Commercial load pattern
        commercial_base = 20
        commercial_pattern = commercial_base * (0.5 + 1.5 * (hour >= 8) * (hour <= 18) * (day_of_week < 5))
        
        # Random events (outages, maintenance, etc.)
        random_events = np.random.poisson(0.01, n_samples) * np.random.normal(-5, 2, n_samples)
        
        # Combine all patterns
        total_load = (daily_pattern + weekly_pattern + seasonal_pattern + 
                     temp_effect + industrial_pattern + residential_pattern + 
                     commercial_pattern + random_events)
        
        # Add noise
        noise = np.random.normal(0, 3, n_samples)
        total_load += noise
        
        # Ensure positive values
        total_load = np.maximum(total_load, 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'load': total_load,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_irradiance': solar_irradiance,
            'electricity_price': electricity_price,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'day_of_year': day_of_year,
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_holiday': holidays,
            'industrial_load': industrial_pattern,
            'residential_load': residential_pattern,
            'commercial_load': commercial_pattern,
            'cooling_load': cooling_effect,
            'heating_load': heating_effect
        })
        
        return df
    
    def generate_holiday_indicators(self, dates):
        """
        Generate holiday indicators for the given dates
        
        Args:
            dates (pd.DatetimeIndex): Date range
            
        Returns:
            np.array: Binary indicators for holidays
        """
        holidays = np.zeros(len(dates))
        
        for i, date in enumerate(dates):
            # Major holidays (simplified)
            if (date.month == 1 and date.day == 1) or \
               (date.month == 7 and date.day == 4) or \
               (date.month == 12 and date.day == 25) or \
               (date.month == 11 and date.day >= 22 and date.day <= 28 and date.dayofweek == 3):
                holidays[i] = 1
        
        return holidays
    
    def add_lag_features(self, data, target_col='load', lags=[1, 2, 3, 24, 48, 168]):
        """
        Add lagged features to the dataset
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            lags (list): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, data, target_col='load', windows=[3, 6, 12, 24]):
        """
        Add rolling statistics features
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            windows (list): List of window sizes
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        df = data.copy()
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def add_time_features(self, data, timestamp_col='timestamp'):
        """
        Add comprehensive time-based features
        
        Args:
            data (pd.DataFrame): Input data
            timestamp_col (str): Timestamp column name
            
        Returns:
            pd.DataFrame: Data with time features
        """
        df = data.copy()
        
        if timestamp_col in df.columns:
            dt = pd.to_datetime(df[timestamp_col])
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
            
            df['day_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            
            df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
            
            df['day_of_year_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
            df['day_of_year_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
            
            # Business day indicators
            df['is_business_day'] = dt.dt.weekday < 5
            df['is_month_start'] = dt.dt.is_month_start
            df['is_month_end'] = dt.dt.is_month_end
            df['is_quarter_start'] = dt.dt.is_quarter_start
            df['is_quarter_end'] = dt.dt.is_quarter_end
        
        return df
    
    def prepare_features(self, data, target_col='load', include_lags=True, 
                        include_rolling=True, include_time=True):
        """
        Prepare comprehensive feature set for modeling
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            include_lags (bool): Whether to include lag features
            include_rolling (bool): Whether to include rolling features
            include_time (bool): Whether to include time features
            
        Returns:
            pd.DataFrame: Processed data with features
        """
        df = data.copy()
        
        if include_time:
            df = self.add_time_features(df)
        
        if include_lags:
            df = self.add_lag_features(df, target_col)
        
        if include_rolling:
            df = self.add_rolling_features(df, target_col)
        
        # Remove rows with NaN values (from lag/rolling features)
        df = df.dropna()
        
        self.features = [col for col in df.columns if col not in ['timestamp', target_col]]
        
        return df
    
    def get_feature_importance_data(self):
        """
        Get data formatted for feature importance analysis
        
        Returns:
            tuple: (X, y, feature_names)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_sample_data() first.")
        
        # Prepare features
        df = self.prepare_features(self.data)
        
        X = df[self.features].values
        y = df['load'].values
        
        return X, y, self.features
    
    def save_data(self, data, file_path):
        """
        Save processed data to CSV
        
        Args:
            data (pd.DataFrame): Data to save
            file_path (str): Output file path
        """
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

def main():
    """
    Demonstration of data loading and feature engineering
    """
    print("Load Data Loader Demonstration")
    print("=" * 40)
    
    # Initialize loader
    loader = LoadDataLoader()
    
    # Generate sample data
    print("Generating comprehensive synthetic data...")
    data = loader.generate_comprehensive_synthetic_data(n_samples=8760)
    print(f"Generated data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Prepare features
    print("\nPreparing features...")
    processed_data = loader.prepare_features(data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Number of features: {len(loader.features)}")
    
    # Display basic statistics
    print("\nLoad statistics:")
    print(data['load'].describe())
    
    # Save sample data
    loader.save_data(processed_data, 'sample_load_data.csv')
    
    print("\nData loading and feature engineering complete!")

if __name__ == "__main__":
    main()