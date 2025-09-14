# Stable Diffusion WebUI Configuration Guide

## Current Setup Analysis

Your current `webui-user.bat` configuration:
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

## 1. Creating a Public Link (Share=True)

To make your WebUI accessible via a public Gradio link, add the `--share` flag to your `COMMANDLINE_ARGS`:

### Option A: Edit webui-user.bat directly
```batch
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --share
```

### Option B: Create a separate public version
Create a new file `webui-user-public.bat`:
```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --share

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

**What happens:** When you run this, Gradio will create a public URL like `https://abc123.gradio.live` that anyone can access.

## 2. Running API-Only Mode

To run only the API without the web interface, use the `--nowebui` flag:

### Option A: API-Only with Public Access
Create `webui-api-only.bat`:
```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --nowebui --share

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

### Option B: API-Only Local
Create `webui-api-local.bat`:
```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --nowebui

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

## 3. Hybrid Mode (WebUI + API)

To run both the web interface AND the API simultaneously, use the `--api` flag:

Create `webui-hybrid.bat`:
```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --api --share

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

## 4. Additional Useful Flags

### Security and Access Control
- `--api-auth username:password` - Add authentication to API
- `--listen` - Listen on all network interfaces (0.0.0.0)
- `--port 7860` - Specify custom port
- `--server-name 0.0.0.0` - Make accessible from other machines on network

### Performance and Logging
- `--api-log` - Enable API request logging
- `--autolaunch` - Automatically open browser
- `--cors-allow-origins http://localhost:3000` - Allow CORS for specific origins

### Example with Security
Create `webui-secure-api.bat`:
```batch
@echo off

set PYTHON=py -3.10
set GIT=
set VENV_DIR=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set WEBUI_LAUNCH_LIVE_OUTPUT=1
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --nowebui --api-auth myuser:mypassword --api-log --port 7861

REM Set QAIRT environment variables
set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
set PATH=%PATH%;C:\Qualcomm\AIStack\QAIRT\2.22.0.240425\bin\x86_64-windows-msvc

REM Activate the virtual environment first
call .\venv\Scripts\activate.bat

call webui.bat
```

## 5. API Endpoints Available

When running with `--api` or `--nowebui`, these endpoints will be available:

- `POST /sdapi/v1/txt2img` - Text to image generation
- `POST /sdapi/v1/img2img` - Image to image generation
- `GET /sdapi/v1/progress` - Get generation progress
- `POST /sdapi/v1/interrupt` - Interrupt current generation
- `GET /sdapi/v1/options` - Get current settings
- `POST /sdapi/v1/options` - Update settings
- `GET /docs` - API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 6. Testing Your Setup

### Test Public Access
1. Run your script with `--share`
2. Look for output like: `Public URL: https://abc123.gradio.live`
3. Share this URL with others

### Test API-Only Mode
1. Run with `--nowebui`
2. Visit `http://localhost:7860/docs` (or your custom port)
3. You should see the API documentation
4. Test with your QAIRT payloads from the sample file

### Test API with Authentication
```bash
curl -X POST "http://localhost:7861/sdapi/v1/txt2img" \
  -H "Authorization: Basic bXl1c2VyOm15cGFzc3dvcmQ=" \
  -H "Content-Type: application/json" \
  -d @sample_api_payloads.json
```

## 7. Recommended Configurations

### For Development (Local + API)
```batch
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --api --api-log
```

### For Production API Server
```batch
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --nowebui --api-auth admin:securepassword --api-log --listen --port 7860
```

### For Public Demo
```batch
set COMMANDLINE_ARGS=--skip-torch-cuda-test --no-half --precision full --ui-config-file .\extensions\qairt_accelerate\ui-config.json --skip-python-version-check --share --api
```

Choose the configuration that best fits your use case!
