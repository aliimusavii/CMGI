"""
Test script for Load Forecasting GUI
Verifies that all components work correctly before building executable
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
        'sklearn',
        'threading',
        'queue',
        'json'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✓ All required modules imported successfully")
        return True

def test_forecasting_systems():
    """Test if forecasting systems work"""
    print("\nTesting forecasting systems...")
    
    # Test simplified system
    try:
        from simple_forecasting import SimpleForecastingSystem
        forecaster = SimpleForecastingSystem()
        data = forecaster.generate_synthetic_data(n_samples=100)
        print("✓ Simplified forecasting system works")
        simple_works = True
    except Exception as e:
        print(f"❌ Simplified system failed: {e}")
        simple_works = False
    
    # Test full system (optional)
    try:
        from load_forecasting_system import LoadForecastingSystem
        forecaster = LoadForecastingSystem()
        data = forecaster.generate_synthetic_data(n_samples=100)
        print("✓ Full forecasting system works")
        full_works = True
    except Exception as e:
        print(f"⚠️ Full system not available: {e}")
        full_works = False
    
    if simple_works:
        print("\n✓ At least one forecasting system is working")
        return True
    else:
        print("\n❌ No forecasting system is working")
        return False

def test_gui_components():
    """Test GUI components without actually showing the window"""
    print("\nTesting GUI components...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Create a test root window (don't show it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test basic widgets
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        button = ttk.Button(frame, text="Test")
        entry = ttk.Entry(frame)
        
        print("✓ Basic tkinter widgets work")
        
        # Test matplotlib with tkinter
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig, ax = plt.subplots(figsize=(4, 3))
        canvas = FigureCanvasTkAgg(fig, frame)
        
        print("✓ Matplotlib with tkinter works")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI components failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'load_forecast_gui.py',
        'simple_forecasting.py',
        'build_exe.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✓ All required files present")
        return True

def run_quick_functionality_test():
    """Run a quick test of core functionality"""
    print("\nRunning quick functionality test...")
    
    try:
        # Test data generation
        from simple_forecasting import SimpleForecastingSystem
        forecaster = SimpleForecastingSystem()
        
        # Generate small dataset
        data = forecaster.generate_synthetic_data(n_samples=200)
        print(f"✓ Generated {len(data)} samples")
        
        # Test sequence preparation
        X, y = forecaster.prepare_sequences(data, features=['load'])
        print(f"✓ Prepared sequences: {X.shape}, {y.shape}")
        
        # Test model building
        model = forecaster.build_random_forest_model(n_estimators=10)
        print("✓ Built Random Forest model")
        
        # Test training (small dataset)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        history = forecaster.train_model(model, X_train, y_train, X_test, y_test, 'Test')
        print("✓ Trained model successfully")
        
        # Test evaluation
        metrics = forecaster.evaluate_model(model, X_test, y_test, 'Test')
        print(f"✓ Model evaluation: RMSE={metrics['RMSE']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("LOAD FORECASTING GUI - PRE-BUILD TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Forecasting Systems Test", test_forecasting_systems),
        ("GUI Components Test", test_gui_components),
        ("Functionality Test", run_quick_functionality_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The system is ready for building the executable.")
        print("\nNext steps:")
        print("1. Run: python build_exe.py")
        print("2. Test the generated executable")
        print("3. Distribute to users")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED!")
        print("Please fix the issues before building the executable.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements_exe.txt")
        print("- Check file paths and permissions")
        print("- Verify Python version compatibility")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)