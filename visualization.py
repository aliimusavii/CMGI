"""
Visualization Tools for Demand Response Forecasting
Creates comprehensive charts and plots for analyzing demand patterns and DR opportunities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class DRVisualization:
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_historical_demand_patterns(self, data, save_path=None):
        """Plot historical demand patterns with DR opportunities highlighted"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Daily demand pattern
        hourly_avg = data.groupby('hour')['demand_mw'].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
        axes[0, 0].axvspan(8, 17, alpha=0.3, color='red', label='DR Target Hours')
        axes[0, 0].set_title('Average Hourly Demand Pattern')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Demand (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weekly demand pattern
        weekly_avg = data.groupby('day_of_week')['demand_mw'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(day_names, weekly_avg.values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Average Daily Demand by Day of Week')
        axes[0, 1].set_ylabel('Demand (MW)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Monthly demand pattern
        monthly_avg = data.groupby('month')['demand_mw'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 0].plot(month_names, monthly_avg.values, marker='s', linewidth=2, color='green')
        axes[1, 0].set_title('Average Monthly Demand Pattern')
        axes[1, 0].set_ylabel('Demand (MW)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Demand vs Temperature
        sample_data = data.sample(min(5000, len(data)))  # Sample for performance
        scatter = axes[1, 1].scatter(sample_data['temperature'], sample_data['demand_mw'], 
                                   alpha=0.6, c=sample_data['hour'], cmap='viridis')
        axes[1, 1].set_title('Demand vs Temperature (colored by hour)')
        axes[1, 1].set_xlabel('Temperature (°F)')
        axes[1, 1].set_ylabel('Demand (MW)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Hour of Day')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dr_forecast_results(self, forecast_results, save_path=None):
        """Plot DR forecast results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # DR Probability by Hour
        hourly_dr_prob = forecast_results.groupby('hour')['dr_probability'].mean()
        axes[0, 0].bar(hourly_dr_prob.index, hourly_dr_prob.values, 
                      color='orange', alpha=0.7)
        axes[0, 0].set_title('Average DR Probability by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('DR Probability')
        axes[0, 0].set_ylim(0, 1)
        
        # DR Events Timeline (first 7 days)
        sample_days = forecast_results.head(7 * 10)  # 7 days * 10 hours (8-17)
        x_pos = range(len(sample_days))
        colors = ['red' if dr else 'blue' for dr in sample_days['dr_recommended']]
        axes[0, 1].scatter(x_pos, sample_days['baseline_demand_mw'], c=colors, alpha=0.7)
        axes[0, 1].set_title('DR Events Timeline (Red=DR, Blue=No DR)')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Baseline Demand (MW)')
        
        # Demand Reduction Impact
        dr_events = forecast_results[forecast_results['dr_recommended'] == 1]
        if len(dr_events) > 0:
            axes[1, 0].hist(dr_events['dr_savings_mw'], bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Distribution of DR Savings')
            axes[1, 0].set_xlabel('DR Savings (MW)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Cumulative DR Impact
        forecast_results['cumulative_savings'] = forecast_results['dr_savings_mw'].cumsum()
        axes[1, 1].plot(forecast_results.index, forecast_results['cumulative_savings'], 
                       color='purple', linewidth=2)
        axes[1, 1].set_title('Cumulative DR Savings Over Time')
        axes[1, 1].set_xlabel('Time Period')
        axes[1, 1].set_ylabel('Cumulative Savings (MW)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dr_dashboard(self, forecast_results, save_path=None):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('DR Events Over Time', 'Hourly DR Probability', 
                          'Demand vs Temperature', 'DR Savings Distribution'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # DR Events Over Time
        fig.add_trace(
            go.Scatter(x=forecast_results['timestamp'], 
                      y=forecast_results['baseline_demand_mw'],
                      mode='lines', name='Baseline Demand',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_results['timestamp'], 
                      y=forecast_results['adjusted_demand_mw'],
                      mode='lines', name='Adjusted Demand',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Add DR events as markers
        dr_events = forecast_results[forecast_results['dr_recommended'] == 1]
        fig.add_trace(
            go.Scatter(x=dr_events['timestamp'], 
                      y=dr_events['baseline_demand_mw'],
                      mode='markers', name='DR Events',
                      marker=dict(color='red', size=8, symbol='diamond')),
            row=1, col=1
        )
        
        # Hourly DR Probability
        hourly_prob = forecast_results.groupby('hour')['dr_probability'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=hourly_prob['hour'], y=hourly_prob['dr_probability'],
                  name='DR Probability', marker_color='orange'),
            row=1, col=2
        )
        
        # Demand vs Temperature
        fig.add_trace(
            go.Scatter(x=forecast_results['temperature'], 
                      y=forecast_results['baseline_demand_mw'],
                      mode='markers', name='Demand vs Temp',
                      marker=dict(color=forecast_results['dr_probability'], 
                                colorscale='Viridis', showscale=True)),
            row=2, col=1
        )
        
        # DR Savings Distribution
        dr_savings = forecast_results[forecast_results['dr_savings_mw'] > 0]['dr_savings_mw']
        fig.add_trace(
            go.Histogram(x=dr_savings, name='DR Savings', 
                        marker_color='green', opacity=0.7),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Demand Response Forecasting Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Demand (MW)", row=1, col=1)
        fig.update_xaxes(title_text="Hour", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_xaxes(title_text="Temperature (°F)", row=2, col=1)
        fig.update_yaxes(title_text="Demand (MW)", row=2, col=1)
        fig.update_xaxes(title_text="DR Savings (MW)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig
    
    def plot_model_performance(self, y_true, y_pred, model_type='classification'):
        """Plot model performance metrics"""
        if model_type == 'classification':
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # ROC Curve (if probabilities available)
            if hasattr(y_pred, 'predict_proba'):
                y_prob = y_pred.predict_proba(y_true)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                
                axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
                axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[1].set_xlim([0.0, 1.0])
                axes[1].set_ylim([0.0, 1.05])
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC Curve')
                axes[1].legend(loc="lower right")
            
        else:  # regression
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Actual vs Predicted
            axes[0].scatter(y_true, y_pred, alpha=0.6)
            axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'r--', lw=2)
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title('Actual vs Predicted')
            
            # Residuals
            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def generate_dr_summary_report(self, forecast_results):
        """Generate summary statistics for DR forecast"""
        total_hours = len(forecast_results)
        dr_hours = forecast_results['dr_recommended'].sum()
        total_savings = forecast_results['dr_savings_mw'].sum()
        avg_savings_per_event = forecast_results[forecast_results['dr_recommended'] == 1]['dr_savings_mw'].mean()
        
        peak_dr_hour = forecast_results.groupby('hour')['dr_recommended'].mean().idxmax()
        peak_dr_prob = forecast_results.groupby('hour')['dr_recommended'].mean().max()
        
        report = f"""
        DEMAND RESPONSE FORECAST SUMMARY
        ================================
        
        Forecast Period: {forecast_results['timestamp'].min()} to {forecast_results['timestamp'].max()}
        Total Hours Analyzed: {total_hours}
        DR Events Recommended: {dr_hours} ({dr_hours/total_hours*100:.1f}%)
        
        SAVINGS ANALYSIS:
        - Total Potential Savings: {total_savings:.1f} MW
        - Average Savings per Event: {avg_savings_per_event:.1f} MW
        - Peak DR Hour: {peak_dr_hour}:00 ({peak_dr_prob*100:.1f}% probability)
        
        HOURLY BREAKDOWN:
        """
        
        hourly_stats = forecast_results.groupby('hour').agg({
            'dr_recommended': ['count', 'sum', 'mean'],
            'dr_savings_mw': 'sum'
        }).round(2)
        
        report += str(hourly_stats)
        
        return report

if __name__ == "__main__":
    # Example usage with sample data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample forecast results
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'hour': dates.hour,
        'baseline_demand_mw': np.random.normal(1000, 200, len(dates)),
        'dr_probability': np.random.random(len(dates)),
        'dr_recommended': np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
        'dr_savings_mw': np.random.normal(50, 15, len(dates)),
        'adjusted_demand_mw': np.random.normal(950, 200, len(dates)),
        'temperature': np.random.normal(75, 15, len(dates))
    })
    
    # Filter for hours 8-17
    sample_data = sample_data[(sample_data['hour'] >= 8) & (sample_data['hour'] <= 17)]
    
    viz = DRVisualization()
    print(viz.generate_dr_summary_report(sample_data))