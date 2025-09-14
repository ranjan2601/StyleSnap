# Virtual Fashion Assistant 👗🤖

Introducing an intelligent revolution in personal style — this project combines **edge AI** and **advanced generative models** to deliver a seamless **Virtual Fashion Assistant**, designed specifically for the **Edge AI Developer Hackathon 2025**.  

Powered by **Sony's Snapdragon X Elite** platform, it harnesses the capabilities of **OpenAI CLIP**, **Anything LLM (Ollama 3.1 8B)**, and **Stable Diffusion** running natively on-device using the **NPU**, thus transforming style discovery, outfit recommendations, and closet management for modern users under **real-time constraints**.  

---

## ✨ Empowering Fashion with Edge AI
Experience the next generation of fashion technology, where **digitized virtual closets** meet **contextual event-based outfit recommendations**.  

By processing all AI tasks **locally on powerful NPUs**, this assistant guarantees:  
- ⚡ Rapid personalization  
- 🔒 Privacy-respecting performance  
- 🌐 Always-available with no cloud dependence  

---

## 🎨 Unleashing Creative Try-Ons and Smart Curation
From digitizing entire wardrobes to suggesting the perfect ensemble for every occasion, this project showcases how AI can unlock **never-before-seen creative styling possibilities**.  

While **fully-quantized virtual try-on with ControlNet** is on the roadmap, our current framework already demonstrates the core pipeline for:  
- 🔍 Intelligent visual search  
- 🎯 Event-aware suggestions  
- 🌀 Interactive exploration of personal style  

Setting a new bar for what’s possible **at the edge**.  

---

## ⚙️ Setup

### Hardware
- **OS:** Windows 11  
- **Chip:** Snapdragon X Elite  
- **Memory:** 32 GB  

### Basic Requirements
- [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)  

---

## 🏗️ Implementation Architecture

Our architecture uses a **3-model stack** to power the fashion recommendation + virtual try-on application.  

### 📌 Stack 1: OpenAI CLIP (Running on NPU)
- We used the **OpenAI CLIP ONNX model**, pretrained and shared by **Qualcomm AI Hub models**.  
- Created a custom implementation of CLIP to run **on-device** with the **Snapdragon X Elite NPU**.  
- Enables **wardrobe image tagging**.  
- Returns sample outputs of **tags with confidence levels**; we pick the top-confidence tags.  
- Exposed through a **FastAPI service**.  

---

## 🛠️ Setup Instructions

```bash
# Create Python environment
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r requirements.txt

# Place ONNX model in models folder
mkdir models && mv your_model.onnx models/

# Run FastAPI service
python fast_clip_service.py
```

### 📌 Stack 2: Stable Diffusion (Running on NPU)

## ⚡ Performance Results
- **NPU Accelerated**: 5-7 seconds per image
- **CPU Fallback**: 30+ seconds per image
- **Supported Models**: Stable Diffusion 1.5, 2.1, ControlNet, ESRGAN upscaling

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

----


### 📌 Stack 3: Llama 3.1 8 B using Anything LLM (Running on NPU) :

Anything LLM setup according to https://github.com/thatrandomfrenchdude/simple-npu-chatbot

Open a PowerShell instance and clone the repo
git clone https://github.com/thatrandomfrenchdude/simple-npu-chatbot.git
Create and activate your virtual environment with reqs
# 1. navigate to the cloned directory
cd simple-npu-chatbot

# 2. create the python virtual environment
python -m venv llm-venv

# 3. activate the virtual environment
./llm-venv/Scripts/Activate.ps1     # windows
source \llm-venv\bin\activate       # mac/linux

# 4. install the requirements
pip install -r requirements.txt
Create your config.yaml file with the following variables
api_key: "your-key-here"
model_server_base_url: "http://localhost:3001/api/v1"
workspace_slug: "your-slug-here"
stream: true
stream_timeout: 60
Test the model server auth to verify the API key
python src/auth.py
Get your workspace slug using the workspaces tool
Run python src/workspaces.py in your command line console
Find your workspace and its slug from the output
Add the slug to the workspace_slug variable in config.yaml
Usage
You have the option to use a terminal or gradio chat interface the talk with the bot. After completing setup, run the app you choose from the command line:

```powershell
# terminal
python src/terminal_chatbot.py
```


## 🚀 Step 7: Launch Backend Services

### 7.1 Start CLIP Service (Port 8001)
```powershell
# Navigate to clip directory
cd clip

# Activate Python environment
python -m venv clip-env
.\clip-env\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Start CLIP FastAPI service
uvicorn fast_clip_service:app --host 0.0.0.0 --port 8001 --reload
```

### 7.2 Start Main Backend API (Port 7000)
```powershell
# Open new PowerShell window
# Navigate to backend directory
cd backend

# Activate Python environment
python -m venv backend-env
.\backend-env\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Initialize SQLite database
python init_db.py

# Start main FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 7000 --reload
```

### 7.3 Verify Backend Services
```powershell
# Test CLIP service
curl http://localhost:8001/health

# Test main backend
curl http://localhost:7000/health

# Check API docs
# CLIP: http://localhost:8001/docs
# Backend: http://localhost:7000/docs
```

---

## 🎨 Step 8: Launch Frontend Application

### 8.1 Install Frontend Dependencies
```powershell
# Open new PowerShell window
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 8.2 Build Frontend Assets
```powershell
# Build production assets
npm run build

# Verify build completed
dir dist
```

### 8.3 Start Development Server
```powershell
# Start Vite development server
npm run dev

# Frontend will be available at:
# http://localhost:5173
```

### 8.4 Alternative: Run Tauri Desktop App
```powershell
# For desktop app experience
npm run tauri:dev

# Or build desktop executable
npm run tauri:build
```

---

## 🔗 Step 9: Service Integration

### 9.1 Verify All Services Running
- ✅ **CLIP Service**: http://localhost:8001
- ✅ **Backend API**: http://localhost:7000  
- ✅ **Frontend**: http://localhost:5173
- ✅ **Stable Diffusion**: http://127.0.0.1:7860

### 9.2 Test Full Pipeline
1. **Upload clothing item** → Backend stores in SQLite
2. **CLIP processes image** → Returns fashion tags
3. **Generate outfit prompt** → Calls Llama 3.1 8B
4. **Create outfit visualization** → Stable Diffusion NPU
5. **Display results** → Frontend shows recommendations

### 9.3 Environment Variables
```powershell
# Set required environment variables
$env:CLIP_SERVICE_URL = "http://localhost:8001"
$env:BACKEND_API_URL = "http://localhost:7000"
$env:STABLE_DIFFUSION_URL = "http://127.0.0.1:7860"
```

---

## 📋 Quick Start Summary

1. **Setup Stable Diffusion** (Steps 1-6) - NPU image generation
2. **Setup Llama 3.1 8B** (Stack 3) - NPU text generation  
3. **Launch Backend** (Step 7) - CLIP + Main API services
4. **Launch Frontend** (Step 8) - React/Tauri application
5. **Test Integration** (Step 9) - Full fashion pipeline

This setup ensures all three AI models (CLIP, Stable Diffusion, Llama 3.1 8B) run on the NPU while the frontend and backend coordinate the fashion styling workflow! 🎨✨

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
