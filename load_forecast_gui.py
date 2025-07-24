"""
Load Forecasting GUI Application
A user-friendly interface for the short-term load forecasting system
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import queue
import os
import sys
from datetime import datetime
import json

# Set matplotlib backend
plt.switch_backend('TkAgg')

class LoadForecastGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Load Forecasting System - Neural Networks & Deep Learning")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.forecaster = None
        self.data = None
        self.models = {}
        self.results = {}
        self.training_thread = None
        self.progress_queue = queue.Queue()
        
        # Create the GUI
        self.create_gui()
        
        # Start progress checker
        self.check_progress()
        
    def create_gui(self):
        """Create the main GUI interface"""
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Load Forecasting System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Right panel - Results and visualization
        self.create_results_panel(main_frame)
        
        # Bottom panel - Progress and logs
        self.create_progress_panel(main_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel with all options"""
        
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Data Generation Section
        data_frame = ttk.LabelFrame(control_frame, text="Data Generation", padding="5")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(data_frame, text="Samples:").grid(row=0, column=0, sticky=tk.W)
        self.samples_var = tk.StringVar(value="2000")
        samples_entry = ttk.Entry(data_frame, textvariable=self.samples_var, width=10)
        samples_entry.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Button(data_frame, text="Generate Data", 
                  command=self.generate_data).grid(row=1, column=0, columnspan=2, pady=5)
        
        self.data_status = ttk.Label(data_frame, text="No data generated", 
                                    foreground="red")
        self.data_status.grid(row=2, column=0, columnspan=2)
        
        # Model Configuration Section
        model_frame = ttk.LabelFrame(control_frame, text="Model Configuration", padding="5")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Sequence Length:").grid(row=0, column=0, sticky=tk.W)
        self.seq_length_var = tk.StringVar(value="24")
        seq_entry = ttk.Entry(model_frame, textvariable=self.seq_length_var, width=10)
        seq_entry.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(model_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="20")
        epochs_entry = ttk.Entry(model_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=1, column=1, padx=(5, 0))
        
        # Model Selection
        ttk.Label(model_frame, text="Select Models:").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        self.model_vars = {}
        # Models available in both systems
        models = ["Random Forest", "Gradient Boosting", "Linear Regression", "SVR", "Ensemble"]
        
        for i, model in enumerate(models):
            var = tk.BooleanVar(value=True if model in ["Random Forest", "Gradient Boosting"] else False)
            self.model_vars[model] = var
            ttk.Checkbutton(model_frame, text=model, variable=var).grid(
                row=3 + i//2, column=i%2, sticky=tk.W, padx=(0, 10))
        
        # Training Section
        train_frame = ttk.LabelFrame(control_frame, text="Training", padding="5")
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.train_button = ttk.Button(train_frame, text="Start Training", 
                                      command=self.start_training)
        self.train_button.grid(row=0, column=0, pady=5)
        
        self.stop_button = ttk.Button(train_frame, text="Stop Training", 
                                     command=self.stop_training, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Results Section
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(results_frame, text="Show Predictions", 
                  command=self.show_predictions).grid(row=0, column=0, pady=2)
        
        ttk.Button(results_frame, text="Compare Models", 
                  command=self.compare_models).grid(row=1, column=0, pady=2)
        
        ttk.Button(results_frame, text="Export Results", 
                  command=self.export_results).grid(row=2, column=0, pady=2)
        
        # Advanced Features
        advanced_frame = ttk.LabelFrame(control_frame, text="Advanced", padding="5")
        advanced_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(advanced_frame, text="Optimize Parameters", 
                  command=self.optimize_parameters).grid(row=0, column=0, pady=2)
        
        ttk.Button(advanced_frame, text="Real-time Simulation", 
                  command=self.real_time_simulation).grid(row=1, column=0, pady=2)
        
    def create_results_panel(self, parent):
        """Create the results and visualization panel"""
        
        results_frame = ttk.LabelFrame(parent, text="Results & Visualization", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results table
        self.create_results_table(results_frame)
        
        # Visualization area
        self.create_visualization_area(results_frame)
        
    def create_results_table(self, parent):
        """Create results table"""
        
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        
        # Create treeview for results
        columns = ('Model', 'RMSE', 'MAE', 'MAPE', 'R²', 'Status')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=6)
        
        # Define headings
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80, anchor='center')
        
        # Scrollbar for table
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def create_visualization_area(self, parent):
        """Create visualization area"""
        
        viz_frame = ttk.Frame(parent)
        viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title("Load Forecasting Results")
        self.ax.set_xlabel("Time (Hours)")
        self.ax.set_ylabel("Load (MW)")
        self.ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Navigation toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def create_progress_panel(self, parent):
        """Create progress and log panel"""
        
        progress_frame = ttk.LabelFrame(parent, text="Progress & Logs", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=0, column=1, padx=(10, 0))
        
        # Log area
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=8, width=80)
        self.log_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def generate_data(self):
        """Generate synthetic load data"""
        try:
            n_samples = int(self.samples_var.get())
            self.log_message(f"Generating {n_samples} samples of synthetic load data...")
            
                         # Import here to handle missing dependencies gracefully
             try:
                 # Try to import the full system first
                 from load_forecasting_system import LoadForecastingSystem
                 self.forecaster = LoadForecastingSystem(
                     sequence_length=int(self.seq_length_var.get()),
                     prediction_horizon=1
                 )
                 self.log_message("Using full neural network system")
             except ImportError:
                 # Fallback to simplified system
                 try:
                     from simple_forecasting import SimpleForecastingSystem
                     self.forecaster = SimpleForecastingSystem(
                         sequence_length=int(self.seq_length_var.get()),
                         prediction_horizon=1
                     )
                     self.log_message("Using simplified system (neural networks not available)")
                 except ImportError as e:
                     raise ImportError("Neither full nor simplified forecasting system available")
                
                self.data = self.forecaster.generate_synthetic_data(n_samples=n_samples)
                
                self.data_status.config(text=f"✓ {len(self.data)} samples generated", 
                                       foreground="green")
                self.log_message(f"Data generated successfully: {len(self.data)} samples")
                self.log_message(f"Load range: {self.data['load'].min():.2f} - {self.data['load'].max():.2f} MW")
                
                # Plot generated data
                self.plot_generated_data()
                
            except ImportError as e:
                messagebox.showerror("Import Error", 
                                   "Required libraries not found. Please ensure all dependencies are installed.")
                self.log_message(f"Import error: {e}")
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number of samples")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
            self.log_message(f"Error generating data: {e}")
            
    def plot_generated_data(self):
        """Plot the generated data"""
        if self.data is not None:
            self.ax.clear()
            # Plot first 168 hours (1 week)
            plot_data = self.data['load'][:168] if len(self.data) > 168 else self.data['load']
            self.ax.plot(plot_data, 'b-', linewidth=1.5, label='Generated Load')
            self.ax.set_title("Generated Load Data (First Week)")
            self.ax.set_xlabel("Time (Hours)")
            self.ax.set_ylabel("Load (MW)")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()
            
    def start_training(self):
        """Start model training in a separate thread"""
        if self.data is None:
            messagebox.showwarning("No Data", "Please generate data first")
            return
        
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("No Models", "Please select at least one model")
            return
        
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.log_message(f"Starting training for models: {', '.join(selected_models)}")
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self.train_models, 
            args=(selected_models,),
            daemon=True
        )
        self.training_thread.start()
        
    def train_models(self, selected_models):
        """Train selected models (runs in separate thread)"""
        try:
            # Prepare data
            self.progress_queue.put(("progress", 10, "Preparing data..."))
            X, y = self.forecaster.prepare_sequences(self.data, features=['load'])
            
            # Split data
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            epochs = int(self.epochs_var.get())
            total_models = len(selected_models)
            
            for i, model_name in enumerate(selected_models):
                if hasattr(threading.current_thread(), '_stop_training'):
                    break
                    
                progress = 20 + (i * 70 // total_models)
                self.progress_queue.put(("progress", progress, f"Training {model_name}..."))
                self.progress_queue.put(("log", f"Training {model_name} model..."))
                
                # Add to results table
                self.progress_queue.put(("result", model_name, "Training..."))
                
                                 try:
                     # Build model (works with both systems)
                     if model_name == "Random Forest":
                         model = self.forecaster.build_random_forest_model()
                     elif model_name == "Gradient Boosting":
                         model = self.forecaster.build_gradient_boosting_model()
                     elif model_name == "Linear Regression":
                         model = self.forecaster.build_linear_model()
                     elif model_name == "SVR":
                         model = self.forecaster.build_svr_model()
                     elif model_name == "Ensemble":
                         model = self.forecaster.build_ensemble_model()
                     else:
                         # Try neural network models if available
                         if hasattr(self.forecaster, 'build_lstm_model'):
                             if model_name == "LSTM":
                                 model = self.forecaster.build_lstm_model(X_train.shape[1:], units=[32])
                             elif model_name == "GRU":
                                 model = self.forecaster.build_gru_model(X_train.shape[1:], units=[32])
                             elif model_name == "CNN-LSTM":
                                 model = self.forecaster.build_cnn_lstm_model(X_train.shape[1:])
                             elif model_name == "Transformer":
                                 model = self.forecaster.build_transformer_model(X_train.shape[1:])
                         else:
                             raise ValueError(f"Model {model_name} not available in simplified system")
                    
                    # Train model
                    history = self.forecaster.train_model(
                        model, X_train, y_train, X_val, y_val, 
                        model_name, epochs=epochs, batch_size=32
                    )
                    
                    # Evaluate model
                    metrics = self.forecaster.evaluate_model(model, X_test, y_test, model_name)
                    
                    # Update results
                    self.progress_queue.put(("result_update", model_name, metrics, "Completed"))
                    self.progress_queue.put(("log", f"{model_name} completed - RMSE: {metrics['RMSE']:.4f}"))
                    
                except Exception as e:
                    self.progress_queue.put(("result_update", model_name, None, f"Error: {str(e)[:20]}"))
                    self.progress_queue.put(("log", f"Error training {model_name}: {e}"))
            
            self.progress_queue.put(("progress", 100, "Training completed"))
            self.progress_queue.put(("log", "All models training completed"))
            self.progress_queue.put(("training_done", None))
            
        except Exception as e:
            self.progress_queue.put(("error", f"Training failed: {str(e)}"))
            self.progress_queue.put(("training_done", None))
            
    def stop_training(self):
        """Stop training"""
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread._stop_training = True
            self.log_message("Training stopped by user")
        
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
    def check_progress(self):
        """Check progress queue and update GUI"""
        try:
            while True:
                message_type, data, *extra = self.progress_queue.get_nowait()
                
                if message_type == "progress":
                    progress, text = data, extra[0] if extra else ""
                    self.progress_var.set(progress)
                    self.progress_label.config(text=text)
                    
                elif message_type == "log":
                    self.log_message(data)
                    
                elif message_type == "result":
                    model_name, status = data, extra[0] if extra else "Training..."
                    self.results_tree.insert('', 'end', values=(model_name, '-', '-', '-', '-', status))
                    
                elif message_type == "result_update":
                    model_name, metrics, status = data, extra[0], extra[1]
                    # Find and update the row
                    for item in self.results_tree.get_children():
                        if self.results_tree.item(item)['values'][0] == model_name:
                            if metrics:
                                values = (model_name, f"{metrics['RMSE']:.4f}", f"{metrics['MAE']:.4f}", 
                                         f"{metrics['MAPE']:.2f}%", f"{metrics['R²']:.4f}", status)
                            else:
                                values = (model_name, '-', '-', '-', '-', status)
                            self.results_tree.item(item, values=values)
                            break
                            
                elif message_type == "training_done":
                    self.train_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    
                elif message_type == "error":
                    messagebox.showerror("Training Error", data)
                    self.train_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_progress)
        
    def show_predictions(self):
        """Show prediction plots"""
        if not hasattr(self.forecaster, 'predictions') or not self.forecaster.predictions:
            messagebox.showwarning("No Results", "Please train models first")
            return
        
        self.ax.clear()
        
        # Plot predictions for all trained models
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, pred_data) in enumerate(self.forecaster.predictions.items()):
            actual = pred_data['actual'][:100]  # First 100 predictions
            predicted = pred_data['predicted'][:100]
            
            if i == 0:  # Plot actual only once
                self.ax.plot(actual, 'k-', linewidth=2, label='Actual', alpha=0.8)
            
            color = colors[i % len(colors)]
            self.ax.plot(predicted, '--', color=color, linewidth=1.5, 
                        label=f'{model_name} Predicted', alpha=0.7)
        
        self.ax.set_title("Load Forecasting Predictions Comparison")
        self.ax.set_xlabel("Time (Hours)")
        self.ax.set_ylabel("Load (MW)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
    def compare_models(self):
        """Show model comparison"""
        if not hasattr(self.forecaster, 'predictions') or not self.forecaster.predictions:
            messagebox.showwarning("No Results", "Please train models first")
            return
        
        # Create comparison window
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Model Comparison")
        comparison_window.geometry("600x400")
        
        # Create comparison table
        columns = ('Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Rank')
        tree = ttk.Treeview(comparison_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor='center')
        
        # Get results and sort by RMSE
        results = []
        for model_name, pred_data in self.forecaster.predictions.items():
            metrics = pred_data['metrics']
            results.append((
                model_name,
                metrics['RMSE'],
                metrics['MAE'],
                metrics['MAPE'],
                metrics['R²']
            ))
        
        # Sort by RMSE (lower is better)
        results.sort(key=lambda x: x[1])
        
        # Add to tree with ranking
        for i, (model, rmse, mae, mape, r2) in enumerate(results):
            tree.insert('', 'end', values=(
                model, f"{rmse:.4f}", f"{mae:.4f}", f"{mape:.2f}", f"{r2:.4f}", i+1
            ))
        
        tree.pack(fill='both', expand=True, padx=10, pady=10)
        
    def export_results(self):
        """Export results to file"""
        if not hasattr(self.forecaster, 'predictions') or not self.forecaster.predictions:
            messagebox.showwarning("No Results", "Please train models first")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    # Export as JSON
                    export_data = {}
                    for model_name, pred_data in self.forecaster.predictions.items():
                        export_data[model_name] = {
                            'metrics': pred_data['metrics'],
                            'predictions': pred_data['predicted'][:100].tolist(),
                            'actual': pred_data['actual'][:100].tolist()
                        }
                    
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                        
                elif filename.endswith('.csv'):
                    # Export as CSV
                    results = []
                    for model_name, pred_data in self.forecaster.predictions.items():
                        metrics = pred_data['metrics']
                        results.append({
                            'Model': model_name,
                            'RMSE': metrics['RMSE'],
                            'MAE': metrics['MAE'],
                            'MAPE': metrics['MAPE'],
                            'R²': metrics['R²']
                        })
                    
                    df = pd.DataFrame(results)
                    df.to_csv(filename, index=False)
                
                messagebox.showinfo("Export Success", f"Results exported to {filename}")
                self.log_message(f"Results exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
                
    def optimize_parameters(self):
        """Run hyperparameter optimization"""
        messagebox.showinfo("Feature", "Hyperparameter optimization will be implemented in the next version")
        
    def real_time_simulation(self):
        """Run real-time simulation"""
        messagebox.showinfo("Feature", "Real-time simulation will be implemented in the next version")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = LoadForecastGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()