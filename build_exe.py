"""
Build script to create standalone executable using PyInstaller
This creates a single executable file with all dependencies included
"""

import os
import sys
import subprocess
import shutil

def build_executable():
    """Build the executable using PyInstaller"""
    
    print("=" * 60)
    print("BUILDING LOAD FORECASTING GUI EXECUTABLE")
    print("=" * 60)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("✓ PyInstaller found")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # No console window (for GUI)
        "--name=LoadForecastingGUI",
        "--add-data=load_forecasting_system.py;.",
        "--add-data=data_loader.py;.",
        "--add-data=advanced_models.py;.",
        "--add-data=README.md;.",
        "--add-data=requirements.txt;.",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=tensorflow",
        "--hidden-import=matplotlib.backends.backend_tkagg",
        "--exclude-module=pytest",
        "--exclude-module=test",
        "--clean",
        "load_forecast_gui.py"
    ]
    
    # Adjust for different platforms
    if sys.platform == "win32":
        cmd.insert(-1, "--icon=NONE")  # You can add an icon file here
    
    print("\n1. Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Build successful!")
            
            # Find the executable
            if sys.platform == "win32":
                exe_path = "dist/LoadForecastingGUI.exe"
            else:
                exe_path = "dist/LoadForecastingGUI"
            
            if os.path.exists(exe_path):
                file_size = os.path.getsize(exe_path) / (1024 * 1024)  # MB
                print(f"✓ Executable created: {exe_path}")
                print(f"✓ File size: {file_size:.1f} MB")
                
                # Create a simple launcher script
                create_launcher_script()
                
                print("\n" + "=" * 60)
                print("BUILD COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print(f"\nExecutable location: {os.path.abspath(exe_path)}")
                print("\nTo run the application:")
                print(f"- Double-click: {exe_path}")
                print("- Or run from command line")
                
                return True
            else:
                print("❌ Executable not found after build")
                return False
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def create_launcher_script():
    """Create a simple launcher script"""
    
    launcher_content = '''@echo off
echo Starting Load Forecasting GUI...
echo.
echo This may take a moment on first run...
echo.

if exist "LoadForecastingGUI.exe" (
    start LoadForecastingGUI.exe
) else (
    echo Error: LoadForecastingGUI.exe not found!
    echo Please ensure the executable is in the same directory.
    pause
)
'''
    
    try:
        with open("dist/run_app.bat", "w") as f:
            f.write(launcher_content)
        print("✓ Launcher script created: dist/run_app.bat")
    except Exception as e:
        print(f"⚠️ Could not create launcher script: {e}")

def create_readme():
    """Create README for the executable"""
    
    readme_content = '''# Load Forecasting GUI - Standalone Executable

## Quick Start

1. Double-click `LoadForecastingGUI.exe` to run the application
2. Or use the launcher: `run_app.bat`

## Features

- Generate synthetic electricity load data
- Train multiple neural network models (LSTM, GRU, CNN-LSTM, Transformer, Ensemble)
- Compare model performance
- Visualize predictions
- Export results

## Usage Instructions

1. **Generate Data**: Click "Generate Data" to create synthetic load data
2. **Configure Models**: Select which models to train and set parameters
3. **Train Models**: Click "Start Training" to begin the training process
4. **View Results**: Use "Show Predictions" and "Compare Models" to analyze results
5. **Export**: Save results to JSON or CSV format

## System Requirements

- Windows 7/8/10/11 (64-bit)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Troubleshooting

If the application doesn't start:
1. Check that you have sufficient memory available
2. Try running as administrator
3. Check the log files in the application directory

## Support

For issues or questions, refer to the README.md file included with the source code.

Built with PyInstaller - All dependencies included.
'''
    
    try:
        with open("dist/README_Executable.txt", "w") as f:
            f.write(readme_content)
        print("✓ Executable README created")
    except Exception as e:
        print(f"⚠️ Could not create executable README: {e}")

def clean_build_files():
    """Clean up build files"""
    
    print("\n2. Cleaning up build files...")
    
    # Directories to remove
    dirs_to_remove = ["build", "__pycache__"]
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✓ Removed {dir_name}/")
            except Exception as e:
                print(f"⚠️ Could not remove {dir_name}/: {e}")
    
    # Files to remove
    files_to_remove = ["LoadForecastingGUI.spec"]
    
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f"✓ Removed {file_name}")
            except Exception as e:
                print(f"⚠️ Could not remove {file_name}: {e}")

def main():
    """Main build function"""
    
    # Check if we're in the right directory
    required_files = ["load_forecast_gui.py", "load_forecasting_system.py"]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            print("Please run this script from the project root directory.")
            return False
    
    # Build the executable
    success = build_executable()
    
    if success:
        # Create additional files
        create_readme()
        
        # Clean up
        clean_build_files()
        
        print("\n🎉 BUILD PROCESS COMPLETED!")
        print("\nNext steps:")
        print("1. Navigate to the 'dist' folder")
        print("2. Run LoadForecastingGUI.exe")
        print("3. Distribute the entire 'dist' folder to users")
        
    else:
        print("\n❌ BUILD FAILED!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()