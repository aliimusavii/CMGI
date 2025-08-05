"""
Demand Response Forecasting Model
Predicts when demand response events should occur during hours 8-17
based on historical demand patterns without DR events.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

class DemandResponseForecaster:
    def __init__(self):
        self.dr_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.demand_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_features(self, data):
        """Prepare features for machine learning models"""
        df = data.copy()
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (previous hours' demand)
        df['demand_lag_1'] = df['demand_mw'].shift(1)
        df['demand_lag_24'] = df['demand_mw'].shift(24)  # Same hour previous day
        df['demand_lag_168'] = df['demand_mw'].shift(168)  # Same hour previous week
        
        # Rolling statistics
        df['demand_rolling_mean_24'] = df['demand_mw'].rolling(window=24, min_periods=1).mean()
        df['demand_rolling_std_24'] = df['demand_mw'].rolling(window=24, min_periods=1).std()
        df['demand_rolling_max_24'] = df['demand_mw'].rolling(window=24, min_periods=1).max()
        
        # Weather interaction features
        df['temp_hour_interaction'] = df['temperature'] * df['hour']
        df['temp_demand_ratio'] = df['temperature'] / (df['demand_mw'] + 1)
        
        # Peak hour indicators
        df['is_peak_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_super_peak'] = ((df['hour'] >= 12) & (df['hour'] <= 16)).astype(int)
        
        # Demand percentile within day
        df['daily_demand_percentile'] = df.groupby(df['timestamp'].dt.date)['demand_mw'].rank(pct=True)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def identify_dr_opportunities(self, data):
        """
        Identify when DR would be most beneficial based on demand patterns
        This simulates the decision-making process for when to implement DR
        """
        df = data.copy()
        
        # Calculate dynamic thresholds for DR opportunities
        df['daily_max_demand'] = df.groupby(df['timestamp'].dt.date)['demand_mw'].transform('max')
        df['daily_mean_demand'] = df.groupby(df['timestamp'].dt.date)['demand_mw'].transform('mean')
        df['weekly_95th_percentile'] = df['demand_mw'].rolling(window=168, min_periods=24).quantile(0.95)
        
        # DR opportunity criteria
        conditions = [
            (df['hour'] >= 8) & (df['hour'] <= 17),  # Must be during target hours
            df['demand_mw'] > df['weekly_95th_percentile'] * 0.9,  # High demand relative to recent history
            df['demand_mw'] > df['daily_mean_demand'] * 1.2,  # Above daily average
            df['temperature'] > 80,  # Hot weather (cooling load)
            ~df['is_weekend'],  # Weekdays more likely for commercial DR
        ]
        
        # Weight the conditions
        weights = [1.0, 0.8, 0.6, 0.4, 0.3]
        df['dr_score'] = sum(condition.astype(int) * weight for condition, weight in zip(conditions, weights))
        
        # Binary DR recommendation (threshold can be tuned)
        df['dr_recommended'] = (df['dr_score'] >= 2.0).astype(int)
        
        return df
    
    def train_models(self, historical_data):
        """Train both classification and regression models"""
        print("Preparing features...")
        df = self.prepare_features(historical_data)
        df = self.identify_dr_opportunities(df)
        
        # Select features for modeling
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
            'demand_rolling_mean_24', 'demand_rolling_std_24', 'demand_rolling_max_24',
            'temperature', 'humidity', 'temp_hour_interaction', 'temp_demand_ratio',
            'is_peak_hour', 'is_super_peak', 'daily_demand_percentile'
        ]
        
        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train DR classification model
        print("Training DR classification model...")
        y_dr = df['dr_recommended']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_dr, test_size=0.2, random_state=42)
        
        self.dr_classifier.fit(X_train, y_train)
        
        # Evaluate classification model
        y_pred = self.dr_classifier.predict(X_test)
        print("DR Classification Results:")
        print(classification_report(y_test, y_pred))
        
        # Train demand regression model (for estimating DR potential savings)
        print("Training demand regression model...")
        y_demand = df['demand_mw']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_demand, test_size=0.2, random_state=42)
        
        self.demand_regressor.fit(X_train_reg, y_train_reg)
        
        # Evaluate regression model
        y_pred_reg = self.demand_regressor.predict(X_test_reg)
        print(f"Demand Regression R²: {r2_score(y_test_reg, y_pred_reg):.3f}")
        print(f"Demand Regression MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f} MW")
        
        self.is_trained = True
        print("Models trained successfully!")
        
        return {
            'classification_accuracy': (y_pred == y_test).mean(),
            'regression_r2': r2_score(y_test_reg, y_pred_reg),
            'regression_mae': mean_absolute_error(y_test_reg, y_pred_reg)
        }
    
    def forecast_dr_events(self, forecast_data, dr_reduction_percent=15):
        """
        Forecast demand response events and their impact
        
        Args:
            forecast_data: DataFrame with future timestamps and weather data
            dr_reduction_percent: Expected demand reduction during DR events (%)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Prepare features for forecasting
        df = forecast_data.copy()
        
        # Add basic demand forecast (without DR)
        df['demand_mw'] = 1000  # Placeholder - would use actual demand forecast
        
        df = self.prepare_features(df)
        
        # Select and scale features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict DR events
        dr_probability = self.dr_classifier.predict_proba(X_scaled)[:, 1]
        dr_binary = self.dr_classifier.predict(X_scaled)
        
        # Predict baseline demand
        baseline_demand = self.demand_regressor.predict(X_scaled)
        
        # Calculate DR impact
        dr_savings = baseline_demand * (dr_reduction_percent / 100) * dr_binary
        adjusted_demand = baseline_demand - dr_savings
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': df['timestamp'],
            'hour': df['hour'],
            'baseline_demand_mw': baseline_demand,
            'dr_probability': dr_probability,
            'dr_recommended': dr_binary,
            'dr_savings_mw': dr_savings,
            'adjusted_demand_mw': adjusted_demand,
            'temperature': df['temperature']
        })
        
        # Filter for hours 8-17 only
        results = results[(results['hour'] >= 8) & (results['hour'] <= 17)]
        
        return results
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        model_data = {
            'dr_classifier': self.dr_classifier,
            'demand_regressor': self.demand_regressor,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.dr_classifier = model_data['dr_classifier']
        self.demand_regressor = model_data['demand_regressor']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_generator import DemandDataGenerator
    
    # Generate historical data
    generator = DemandDataGenerator()
    historical_data = generator.generate_historical_data()
    
    # Train model
    forecaster = DemandResponseForecaster()
    metrics = forecaster.train_models(historical_data)
    
    print(f"Training completed with metrics: {metrics}")
    
    # Save model
    forecaster.save_model('/workspace/dr_model.pkl')