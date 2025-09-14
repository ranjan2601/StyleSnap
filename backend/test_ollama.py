#!/usr/bin/env python3
"""
Test script to verify Ollama connection
"""
import asyncio
import httpx
import json

# Configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

async def test_ollama_connection():
    """Test the connection to Ollama"""
    try:
        print("Testing Ollama connection...")
        print(f"Base URL: {OLLAMA_BASE_URL}")
        print(f"Model: {OLLAMA_MODEL}")
        
        async with httpx.AsyncClient() as client:
            # Test with a simple message
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "Hello! Please respond with 'Connection successful' if you can read this.",
                    "stream": False
                },
                timeout=30.0
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Connection successful!")
                print(f"Response: {result.get('response', 'No response')}")
                return True
            else:
                print(f"❌ Connection failed!")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

async def test_outfit_generation():
    """Test outfit generation with sample data"""
    try:
        print("\nTesting outfit generation...")
        
        # Sample wardrobe data
        wardrobe_data = [
            {
                "image_path": "uploads/test1.jpg",
                "tags": {"color": "blue", "category": "shirt", "style": "casual"}
            },
            {
                "image_path": "uploads/test2.jpg", 
                "tags": {"color": "black", "category": "pants", "style": "formal"}
            },
            {
                "image_path": "uploads/test3.jpg",
                "tags": {"color": "white", "category": "shoes", "style": "casual"}
            }
        ]
        
        user_input = "casual outfit for brunch"
        
        prompt = f"""You are a fashion AI assistant. Based on the following wardrobe data, suggest an outfit for: "{user_input}"

Wardrobe Data:
{json.dumps(wardrobe_data, indent=2)}

Please analyze the wardrobe and suggest 2-4 items that would make a good outfit for the occasion. 
Return ONLY a JSON array of image paths (no other text, no explanations).

Example format: ["uploads/test1.jpg", "uploads/test2.jpg"]"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')
                print(f"✅ Outfit generation test successful!")
                print(f"LLM Response: {llm_response}")
                
                # Try to parse the JSON response
                try:
                    import re
                    json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
                    if json_match:
                        image_paths = json.loads(json_match.group())
                        print(f"✅ Parsed JSON successfully: {image_paths}")
                        return True
                    else:
                        print(f"⚠️  Could not find JSON array in response")
                        return False
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {str(e)}")
                    return False
            else:
                print(f"❌ Outfit generation failed!")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Outfit generation error: {str(e)}")
        return False

async def test_model_availability():
    """Test if the model is available"""
    try:
        print("\nTesting model availability...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                print(f"Available models: {model_names}")
                
                if OLLAMA_MODEL in model_names:
                    print(f"✅ Model {OLLAMA_MODEL} is available!")
                    return True
                else:
                    print(f"❌ Model {OLLAMA_MODEL} not found!")
                    print(f"Available models: {model_names}")
                    return False
            else:
                print(f"❌ Failed to get model list: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Model availability check error: {str(e)}")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 Testing Ollama Integration")
        print("=" * 50)
        
        # Test model availability
        model_ok = await test_model_availability()
        
        if model_ok:
            # Test basic connection
            connection_ok = await test_ollama_connection()
            
            if connection_ok:
                # Test outfit generation
                await test_outfit_generation()
        
        print("\n" + "=" * 50)
        if model_ok:
            print("🎉 Ollama is ready to use!")
        else:
            print("⚠️  Please check your Ollama setup:")
            print("   1. Make sure Ollama is running: ollama serve")
            print(f"   2. Pull the model: ollama pull {OLLAMA_MODEL}")
            print("   3. Check if Ollama is accessible at http://127.0.0.1:11434")
    
    asyncio.run(main())
