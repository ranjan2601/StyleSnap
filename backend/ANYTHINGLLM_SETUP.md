# AnythingLLM Setup Guide

## 🚀 How to Connect AnythingLLM to StyleSnap

### Step 1: Install AnythingLLM

1. **Download AnythingLLM**:
   - Go to [https://anythingllm.com/](https://anythingllm.com/)
   - Download the appropriate version for your operating system
   - Install and run AnythingLLM

2. **Start AnythingLLM**:
   - Launch the AnythingLLM application
   - It should start on `http://localhost:3001` by default
   - If it uses a different port, note it down

### Step 2: Configure AnythingLLM

1. **Open AnythingLLM Settings**:
   - Click the **Settings** icon (gear/wrench) in the bottom left
   - Go to **LLM API Providers**

2. **Set up LLM Provider**:
   - Select **Generic OpenAI** as the provider
   - **API Key**: `CTKMGJ8-32Y4D5G-H6BNNKA-DGA0TQ8`
   - **Base URL**: Enter your LLM service URL (e.g., OpenAI, Anthropic, etc.)
   - **Model ID**: Enter the model you want to use (e.g., `gpt-3.5-turbo`, `claude-3-sonnet`)

3. **Create a Workspace**:
   - Go to **Workspaces** in the sidebar
   - Create a new workspace or use the default one
   - Note the workspace ID (usually "default" or a UUID)

### Step 3: Update Configuration

If AnythingLLM is running on a different port or URL, update the configuration in `main.py`:

```python
# Update these values in main.py
ANYTHINGLLM_BASE_URL = "http://localhost:3001"  # Change if different
ANYTHINGLLM_WORKSPACE_ID = "default"  # Change to your workspace ID
```

### Step 4: Test the Connection

Run the test script to verify everything is working:

```bash
cd StyleSnap/backend
python test_anythingllm.py
```

### Step 5: Start the Backend

```bash
cd StyleSnap/backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 🔧 Troubleshooting

### Common Issues:

1. **"Could not find working AnythingLLM endpoint"**:
   - Make sure AnythingLLM is running
   - Check if it's on a different port (try 3000, 3001, 8080)
   - Verify the workspace ID is correct

2. **"API Key invalid"**:
   - Double-check the API key: `CTKMGJ8-32Y4D5G-H6BNNKA-DGA0TQ8`
   - Make sure it's configured in AnythingLLM settings

3. **"Workspace not found"**:
   - Create a workspace in AnythingLLM
   - Update the `ANYTHINGLLM_WORKSPACE_ID` in `main.py`

### Alternative Setup (If AnythingLLM doesn't work):

The app will automatically fall back to a simple rule-based outfit selection if AnythingLLM is not available. This ensures the app still works while you set up the LLM integration.

## 📝 API Endpoints

The app will try these endpoints automatically:
- `/api/workspace/chat`
- `/api/workspace/stream-chat`
- `/api/chat`
- `/api/v1/chat`
- `/api/workspace/chat-stream`
- `/api/workspace/stream`

## 🎯 How It Works

1. User enters a prompt (e.g., "casual brunch outfit")
2. Backend queries the SQLite database for all wardrobe items
3. Backend sends wardrobe data + user prompt to AnythingLLM
4. AnythingLLM analyzes the data and returns JSON array of image paths
5. Frontend displays the selected outfit items

## 📞 Support

If you're still having issues:
1. Check the console logs for error messages
2. Verify AnythingLLM is running and accessible
3. Test the connection with the provided test script
4. The app will work with fallback mode if AnythingLLM is unavailable
