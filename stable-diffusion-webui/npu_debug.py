import sys
import os
import onnxruntime as ort

def check_system_info():
    """Check basic system information"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    print()

def check_onnxruntime_providers():
    """Check available ONNX Runtime providers"""
    print("=== ONNX Runtime Providers ===")
    try:
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        # Check specifically for QNN (Qualcomm Neural Network) provider
        qnn_providers = [p for p in providers if 'QNN' in p.upper() or 'QUALCOMM' in p.upper()]
        if qnn_providers:
            print(f"✅ QNN/Qualcomm providers found: {qnn_providers}")
        else:
            print("❌ No QNN/Qualcomm providers found")
            
        # Check for CPU provider
        cpu_providers = [p for p in providers if 'CPU' in p.upper()]
        if cpu_providers:
            print(f"CPU providers: {cpu_providers}")
            
    except Exception as e:
        print(f"Error checking ONNX Runtime providers: {e}")
    print()

def check_qairt_installation():
    """Check QAIRT installation"""
    print("=== QAIRT Installation Check ===")
    try:
        import qai_appbuilder
        print(f"✅ qai-appbuilder version: {qai_appbuilder.__version__}")
    except ImportError as e:
        print(f"❌ qai-appbuilder not found: {e}")
    
    # Check for QAIRT SDK files
    qairt_paths = [
        "extensions/qairt_accelerate/qnn_assets/qnn_libs",
        "extensions/qairt_accelerate/qnn_assets/cache"
    ]
    
    for path in qairt_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            files = os.listdir(path)
            print(f"   Files: {files[:5]}...")  # Show first 5 files
        else:
            print(f"❌ Missing: {path}")
    print()

def check_model_files():
    """Check for NPU model files"""
    print("=== NPU Model Files Check ===")
    model_paths = [
        "extensions/qairt_accelerate/qnn_assets/cache",
        "models/Stable-diffusion"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            files = os.listdir(path)
            # Look for .bin files (NPU models) and .safetensors files
            bin_files = [f for f in files if f.endswith('.bin')]
            safetensors_files = [f for f in files if f.endswith('.safetensors')]
            
            if bin_files:
                print(f"   NPU model files (.bin): {bin_files}")
            if safetensors_files:
                print(f"   Standard model files (.safetensors): {safetensors_files}")
        else:
            print(f"❌ Missing: {path}")
    print()

def test_npu_inference():
    """Test NPU inference capability"""
    print("=== NPU Inference Test ===")
    try:
        # Try to create a simple ONNX Runtime session with QNN provider
        providers = ort.get_available_providers()
        qnn_providers = [p for p in providers if 'QNN' in p.upper() or 'QUALCOMM' in p.upper()]
        
        if qnn_providers:
            print(f"Attempting to use QNN provider: {qnn_providers[0]}")
            # This is a basic test - in reality you'd need a proper ONNX model
            print("✅ QNN provider is available for inference")
        else:
            print("❌ No QNN provider available - will fallback to CPU")
            
    except Exception as e:
        print(f"Error testing NPU inference: {e}")
    print()

def main():
    print("🔍 NPU Detection Debug Script")
    print("=" * 50)
    
    check_system_info()
    check_onnxruntime_providers()
    check_qairt_installation()
    check_model_files()
    test_npu_inference()
    
    print("=" * 50)
    print("Debug script completed!")

if __name__ == "__main__":
    main()
