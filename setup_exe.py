"""
Setup script to create standalone executable for Load Forecasting GUI
Uses cx_Freeze to bundle all dependencies
"""

import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but some modules need to be explicitly included
build_exe_options = {
    "packages": [
        "tkinter", "matplotlib", "numpy", "pandas", "sklearn", "tensorflow",
        "queue", "threading", "json", "datetime", "os", "sys"
    ],
    "excludes": [
        "test", "unittest", "email", "html", "http", "urllib", "xml",
        "pydoc", "doctest", "argparse", "pickle"
    ],
    "include_files": [
        "load_forecasting_system.py",
        "data_loader.py", 
        "advanced_models.py",
        "README.md",
        "requirements.txt"
    ],
    "optimize": 2,
    "build_exe": "dist/LoadForecastingGUI"
}

# GUI applications on Windows should use "Win32GUI" base
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Load Forecasting System",
    version="1.0",
    description="Neural Network Load Forecasting with GUI",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "load_forecast_gui.py",
            base=base,
            target_name="LoadForecastingGUI.exe" if sys.platform == "win32" else "LoadForecastingGUI",
            icon=None  # You can add an icon file here
        )
    ]
)