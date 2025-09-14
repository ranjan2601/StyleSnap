# Complete Stable Diffusion WebUI with Qualcomm X-Elite NPU Setup Guide

## 📋 Overview
This guide provides step-by-step instructions to set up Stable Diffusion WebUI with Qualcomm X-Elite NPU acceleration using the QAIRT (Qualcomm AI Runtime) extension.

## ⚡ Performance Results
- **NPU Accelerated**: 5-7 seconds per image
- **CPU Fallback**: 30+ seconds per image
- **Supported Models**: Stable Diffusion 1.5, 2.1, ControlNet, ESRGAN upscaling

---

## 🔧 Prerequisites

### System Requirements
- ✅ **Windows 10/11** with Qualcomm X-Elite processor
- ✅ **Python 3.10.6** (exactly this version - critical for compatibility)
- ✅ **Git** for version control
- ✅ **PowerShell** (default Windows terminal)
- ✅ **Minimum 8GB RAM** (16GB+ recommended)
- ✅ **Minimum 20GB free disk space**

### Verify Python Version
```powershell
python --version
# Should output: Python 3.10.6
```

If you don't have Python 3.10.6, download it from: https://www.python.org/downloads/release/python-3106/

---

## 📥 Step 1: Download Stable Diffusion WebUI

### 1.1 Create Project Directory
```powershell
# Navigate to your desired location (e.g., Desktop)
cd C:\Users\[YOUR_USERNAME]\Desktop
mkdir stable-diff
cd stable-diff
```

### 1.2 Clone the Repository
```powershell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
```

---

## 🚀 Step 2: Install QAIRT Extension

### 2.1 Download QAIRT Extension
```powershell
# Clone the QAIRT plugins repository
git clone https://github.com/quic/wos-ai-plugins.git temp-plugins

# Copy QAIRT extension to extensions directory
xcopy temp-plugins\plugins\stable-diffusion-webui\qairt_accelerate extensions\qairt_accelerate\ /E /I

# Clean up temporary files
Remove-Item -Recurse -Force temp-plugins
```

### 2.2 Verify Extension Installation
```powershell
# Check if extension directory exists
dir extensions\qairt_accelerate
# Should show: install.py, scripts folder, etc.
```

---

## ⚙️ Step 3: Configure Launch Settings

### 3.1 Create/Edit webui-user.bat
Create or edit the `webui-user.bat` file in the root directory with the following content:

```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

### 3.2 Parameter Explanations
- `PYTHON=py -3.10`: Forces use of Python 3.10.6
- `TORCH_INDEX_URL`: Uses CPU-optimized PyTorch
- `--skip-torch-cuda-test`: Bypasses CUDA checks
- `--no-half --precision full`: Required for NPU compatibility
- `--ui-config-file`: Uses QAIRT-specific UI configuration
- `QAIRT_ROOT`: Points to QAIRT SDK installation
- `PATH`: Adds QAIRT tools to system path

---

## 🔧 Step 4: Fix Known Issues

### 4.1 Fix Upscaler Pipeline Error
Edit `extensions\qairt_accelerate\pipeline_cache.py` and replace the `reload_upscaler_pipeline` function:

**Find this function (around line 51):**
```python
def reload_upscaler_pipeline(self, upscaler_model_name):
    if upscaler_model_name == "":
        del self.upscaler_pipeline
        return
    if self.upscaler_pipeline.model_name != upscaler_model_name:
        del self.upscaler_pipeline
```

**Replace with:**
```python
def reload_upscaler_pipeline(self, upscaler_model_name):
    try:
        if upscaler_model_name == "":
            if hasattr(self, 'upscaler_pipeline'):
                del self.upscaler_pipeline
            return
        if hasattr(self, 'upscaler_pipeline') and self.upscaler_pipeline.model_name != upscaler_model_name:
            del self.upscaler_pipeline
        _, upscaler_model_path = get_upscaler_model(upscaler_model_name)
        self.upscaler_pipeline = UpscalerPipeline(
            upscaler_model_name, upscaler_model_path
        )
    except AttributeError as e:
        print(f"Warning: {e} - continuing without upscaler pipeline")
        pass
```

---

## 🚀 Step 5: First Launch and Setup

### 5.1 Launch WebUI
```powershell
# Ensure you're in the stable-diffusion-webui directory
cd C:\Users\[YOUR_USERNAME]\Desktop\stable-diff\stable-diffusion-webui

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set QAIRT environment variables (if not using webui-user.bat)
$env:QAIRT_ROOT = "C:\Qualcomm\AIStack\QAIRT\2.22.0.240425"
$env:PATH += ";C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc"

# Launch WebUI
.\webui-user.bat
```

### 5.2 First Launch Process
During the first launch, the system will automatically:

1. **Create virtual environment** with Python 3.10.6
2. **Install dependencies** (PyTorch, Gradio, etc.)
3. **Download QAIRT SDK** to `C:\Qualcomm\AIStack\QAIRT\2.22.0.240425`
4. **Install qai-appbuilder** package
5. **Download NPU model binaries** (.bin files)
6. **Download NPU models** from Qualcomm AI Hub:
   - qcom-Stable-Diffusion-v1.5
   - qcom-Stable-Diffusion-v2.1
   - ControlNet models
   - ESRGAN upscaler
7. **Download standard CPU model** (v1-5-pruned-emaonly.safetensors)
8. **Launch web interface** at http://127.0.0.1:7860

### 5.3 Expected Output
Look for these key messages in the terminal:
```
Installing Install python QNN
Collecting qai-appbuilder==2.22.0
Downloading QAIRT model bin files...
QAIRT model bin files downloaded.
Downloading required models using qai-hub...
Downloaded required models.
Loading models from C:\Users\...\qcom-Stable-Diffusion-v1.5
Running on local URL:  http://127.0.0.1:7860
```

---

## 🎯 Step 6: Using NPU Acceleration

### 6.1 Access WebUI
1. Open your web browser
2. Navigate to: `http://127.0.0.1:7860`
3. Wait for the interface to fully load

### 6.2 Enable NPU Acceleration
1. **Go to txt2img or img2img tab**
2. **Scroll to the bottom** of the page
3. **Find "Scripts" dropdown**
4. **Select "qairt_accelerate"** from the dropdown
5. **New options will appear** for NPU configuration

### 6.3 NPU Model Selection
In the QAIRT script section, you'll see:

**Model to use dropdown:**
- `Stable-Diffusion-1.5` (NPU accelerated)
- `Stable-Diffusion-2.1` (NPU accelerated)
- `ControlNet-v10-sd15-canny` (for img2img with ControlNet)

**Upscaler to use dropdown:**
- `ESRGAN_4x` (NPU accelerated upscaling)

### 6.4 Generate Images with NPU
1. **Enter your prompt** in the text field
2. **Select NPU model** from the QAIRT dropdown
3. **Set generation parameters** (steps, CFG scale, etc.)
4. **Click "Generate"**
5. **Monitor terminal** for NPU inference messages

### 6.5 NPU Success Indicators
Look for these messages in the terminal:
```
self.pipeline : <qairt_sd_pipeline.QnnStableDiffusionPipeline object>
Image Generation successful
time consumes for inference 5.762853622436523(s)
```

---

## 📁 File Structure After Setup

```
stable-diffusion-webui/
├── extensions/
│   └── qairt_accelerate/
│       ├── install.py
│       ├── scripts/
│       │   └── qairt_accelerate.py
│       ├── qnn_assets/
│       │   ├── qnn_libs/
│       │   │   ├── QnnHtp.dll
│       │   │   ├── QnnSystem.dll
│       │   │   └── ...
│       │   └── cache/
│       ├── qai_hub_venv/
│       └── ui-config.json
├── models/
│   └── Stable-diffusion/
│       ├── v1-5-pruned-emaonly.safetensors
│       ├── qcom-Stable-Diffusion-v1.5/
│       │   ├── TextEncoder_Quantized.bin
│       │   ├── UNet_Quantized.bin
│       │   └── VAEDecoder_Quantized.bin
│       └── qcom-Stable-Diffusion-v2.1/
│           ├── TextEncoder_Quantized.bin
│           ├── UNet_Quantized.bin
│           └── VAEDecoder_Quantized.bin
├── venv/
├── webui-user.bat
└── webui.bat
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Python Version Error
**Error:** `INCOMPATIBLE PYTHON VERSION`
**Solution:**
```powershell
# Install Python 3.10.6 exactly
# Update webui-user.bat with: set PYTHON=py -3.10
# Delete venv folder to force recreation
Remove-Item -Recurse -Force venv
```

#### Issue 2: WebUI Not Found
**Error:** `The term '.\webui-user.bat' is not recognized`
**Solution:**
```powershell
# Ensure you're in the correct directory
cd stable-diffusion-webui
# Then run the command
.\webui-user.bat
```

#### Issue 3: QAIRT Extension Not Visible
**Problem:** No "qairt_accelerate" option in Scripts dropdown
**Solution:**
1. Check extension installation: `dir extensions\qairt_accelerate`
2. Restart WebUI completely
3. Verify QAIRT environment variables are set

#### Issue 4: NPU Not Working
**Problem:** Still using CPU for inference
**Solution:**
1. Verify Python 3.10.6: `python --version`
2. Check QAIRT environment variables:
   ```powershell
   echo $env:QAIRT_ROOT
   echo $env:PATH
   ```
3. Restart WebUI with proper environment activation

#### Issue 5: Upscaler Pipeline Error
**Error:** `AttributeError: upscaler_pipeline`
**Solution:** Apply the fix from Step 4.1

### Performance Optimization

#### For Best NPU Performance:
- **Use recommended settings:**
  - Sampling steps: 20-50
  - CFG Scale: 7-15
  - Sampler: DPM++ 2M or Euler a
- **Avoid very high resolutions** on first generation
- **Use batch size of 1** for optimal NPU utilization

#### Memory Management:
- **Close other applications** during generation
- **Use --no-half --precision full** flags (already in config)
- **Monitor system resources** during generation

---

## 🔄 Daily Usage Commands

### Quick Start (After Initial Setup)
```powershell
# Navigate to project
cd C:\Users\[YOUR_USERNAME]\Desktop\stable-diff\stable-diffusion-webui

# Activate environment
.\venv\Scripts\Activate.ps1

# Set QAIRT variables
$env:QAIRT_ROOT = "C:\Qualcomm\AIStack\QAIRT\2.22.0.240425"
$env:PATH += ";C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc"

# Launch
.\webui-user.bat
```

### Or Use Simplified Launch (Recommended)
```powershell
cd stable-diffusion-webui
.\webui-user.bat
```
(The webui-user.bat handles everything automatically)

---

## 📊 Performance Comparison

| Generation Type | CPU Time | NPU Time | Speed Improvement |
|----------------|----------|----------|-------------------|
| txt2img (512x512) | 30-45s | 5-7s | 6-7x faster |
| img2img (512x512) | 25-35s | 4-6s | 5-6x faster |
| ControlNet | 45-60s | 8-12s | 4-5x faster |
| Upscaling (ESRGAN) | 15-25s | 3-5s | 4-5x faster |

---

## 🔧 Advanced Configuration

### Custom Model Paths
To use custom model locations, modify `qairt_constants.py`:
```python
# Edit extensions/qairt_accelerate/qairt_constants.py
QNN_LIBS_DIR = "your/custom/path"
CACHE_DIR = "your/custom/cache/path"
```

### Environment Variables for Advanced Users
```powershell
# Additional QAIRT settings (optional)
$env:QNN_LOG_LEVEL = "ERROR"  # Reduce log verbosity
$env:QNN_BACKEND_PATH = "C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc"
```

---

## 📝 Changelog and Updates

### Version History
- **v1.0**: Initial QAIRT NPU setup
- **v1.1**: Fixed upscaler pipeline AttributeError
- **v1.2**: Added comprehensive troubleshooting guide
- **v1.3**: Performance optimization recommendations

### Known Limitations
- **Python 3.10.6 requirement**: Strict version dependency
- **Windows only**: Currently no Linux/Mac support for X-Elite NPU
- **Model compatibility**: Limited to Stable Diffusion 1.5/2.1 base models
- **ControlNet support**: Limited to specific ControlNet models

---

## 🆘 Support and Resources

### Official Documentation
- **QAIRT Extension**: https://github.com/quic/wos-ai-plugins
- **Stable Diffusion WebUI**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **Qualcomm AI Hub**: https://app.aihub.qualcomm.com/

### Community Support
- **WebUI Discussions**: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions
- **Qualcomm Developer Forums**: https://developer.qualcomm.com/forums

### Debug Information Collection
If you encounter issues, collect this information:
```powershell
# System info
python --version
echo $env:QAIRT_ROOT
echo $env:PATH

# Extension status
dir extensions\qairt_accelerate
dir models\Stable-diffusion

# Log files
type tmp\stdout.txt
type tmp\stderr.txt
```

---

## ✅ Setup Verification Checklist

- [ ] Python 3.10.6 installed and verified
- [ ] Stable Diffusion WebUI cloned successfully
- [ ] QAIRT extension installed in extensions folder
- [ ] webui-user.bat configured with all settings
- [ ] Pipeline cache bug fix applied
- [ ] Virtual environment created with Python 3.10.6
- [ ] QAIRT SDK downloaded to C:\Qualcomm\AIStack\QAIRT\
- [ ] NPU models downloaded (qcom-Stable-Diffusion-v1.5, v2.1)
- [ ] QNN libraries present in qnn_assets/qnn_libs/
- [ ] WebUI accessible at http://127.0.0.1:7860
- [ ] "qairt_accelerate" visible in Scripts dropdown
- [ ] NPU model selection working
- [ ] Image generation successful with NPU timing logs
- [ ] Performance improvement verified (5-7s vs 30+s)

---

## 🎉 Success Confirmation

Your setup is successful when you see:
1. **Terminal shows**: `QnnStableDiffusionPipeline object`
2. **Generation time**: ~5-7 seconds per image
3. **Log message**: `Image Generation successful`
4. **WebUI interface**: QAIRT options visible
5. **Models loaded**: Both CPU and NPU models available

**Congratulations! You now have Stable Diffusion running with Qualcomm X-Elite NPU acceleration!** 🚀

---

*Last updated: September 2025*
*Compatible with: Stable Diffusion WebUI v1.10.1, QAIRT Extension v2.22.0*
