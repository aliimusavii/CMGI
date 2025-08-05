"""
Historical Demand Data Generator
Generates realistic electrical demand data without demand response events
for training the DR forecasting model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

class DemandDataGenerator:
    def __init__(self, start_date='2023-01-01', end_date='2024-01-01'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.holidays = holidays.US()
        
    def generate_base_load_pattern(self, hour, day_of_week, month, is_holiday):
        """Generate base load pattern based on time factors"""
        # Base load varies by hour (typical daily pattern)
        hourly_pattern = {
            0: 0.6, 1: 0.55, 2: 0.52, 3: 0.50, 4: 0.52, 5: 0.58,
            6: 0.68, 7: 0.82, 8: 0.95, 9: 1.0, 10: 1.02, 11: 1.05,
            12: 1.08, 13: 1.10, 14: 1.12, 15: 1.15, 16: 1.18, 17: 1.15,
            18: 1.10, 19: 1.05, 20: 0.98, 21: 0.88, 22: 0.78, 23: 0.68
        }
        
        # Weekend vs weekday adjustment
        weekend_factor = 0.85 if day_of_week >= 5 else 1.0
        
        # Holiday adjustment
        holiday_factor = 0.75 if is_holiday else 1.0
        
        # Seasonal adjustment
        seasonal_factors = {
            1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.90, 6: 1.20,
            7: 1.25, 8: 1.22, 9: 1.10, 10: 1.00, 11: 1.05, 12: 1.12
        }
        
        base_load = 1000  # Base MW
        return base_load * hourly_pattern[hour] * weekend_factor * holiday_factor * seasonal_factors[month]
    
    def add_weather_impact(self, base_demand, month, hour):
        """Add weather-related demand variations"""
        # Summer cooling load (hours 12-18)
        if month in [6, 7, 8] and 12 <= hour <= 18:
            weather_factor = np.random.normal(1.3, 0.2)
        # Winter heating load (hours 6-9, 17-21)
        elif month in [12, 1, 2] and (6 <= hour <= 9 or 17 <= hour <= 21):
            weather_factor = np.random.normal(1.2, 0.15)
        else:
            weather_factor = np.random.normal(1.0, 0.1)
            
        return base_demand * max(0.5, weather_factor)
    
    def add_random_variations(self, demand):
        """Add random variations to make data realistic"""
        noise = np.random.normal(0, demand * 0.05)  # 5% random variation
        return max(0, demand + noise)
    
    def generate_historical_data(self):
        """Generate complete historical demand dataset"""
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='H')[:-1]
        
        data = []
        for timestamp in date_range:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            month = timestamp.month
            is_holiday = timestamp.date() in self.holidays
            
            # Generate base demand
            base_demand = self.generate_base_load_pattern(hour, day_of_week, month, is_holiday)
            
            # Add weather impact
            weather_demand = self.add_weather_impact(base_demand, month, hour)
            
            # Add random variations
            final_demand = self.add_random_variations(weather_demand)
            
            # Create potential DR opportunity indicator (high demand periods)
            # This will be used to identify when DR would be beneficial
            dr_potential = 1 if (hour >= 8 and hour <= 17 and 
                               final_demand > np.percentile([self.generate_base_load_pattern(h, day_of_week, month, is_holiday) 
                                                           for h in range(24)], 75)) else 0
            
            data.append({
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': day_of_week >= 5,
                'is_holiday': is_holiday,
                'demand_mw': round(final_demand, 2),
                'dr_potential': dr_potential,  # Indicator for when DR would be beneficial
                'temperature': self.generate_temperature(month),
                'humidity': np.random.uniform(30, 90)
            })
        
        return pd.DataFrame(data)
    
    def generate_temperature(self, month):
        """Generate realistic temperature data"""
        temp_ranges = {
            1: (25, 45), 2: (30, 50), 3: (40, 60), 4: (50, 70),
            5: (60, 80), 6: (70, 90), 7: (75, 95), 8: (75, 95),
            9: (65, 85), 10: (55, 75), 11: (40, 60), 12: (25, 45)
        }
        min_temp, max_temp = temp_ranges[month]
        return round(np.random.uniform(min_temp, max_temp), 1)

if __name__ == "__main__":
    # Generate sample data
    generator = DemandDataGenerator()
    historical_data = generator.generate_historical_data()
    
    # Save to CSV
    historical_data.to_csv('/workspace/historical_demand_data.csv', index=False)
    print(f"Generated {len(historical_data)} hours of historical demand data")
    print(f"Data saved to historical_demand_data.csv")
    print(f"Sample data:\n{historical_data.head()}")