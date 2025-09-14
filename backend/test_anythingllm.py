#!/usr/bin/env python3
"""
Test script to verify AnythingLLM connection
"""
import asyncio
import httpx
import json

# Configuration
ANYTHINGLLM_API_KEY = "CTKMGJ8-32Y4D5G-H6BNNKA-DGA0TQ8"
ANYTHINGLLM_BASE_URL = "http://localhost:3001"  # Default AnythingLLM URL
ANYTHINGLLM_WORKSPACE_ID = "default"

async def test_anythingllm_connection():
    """Test the connection to AnythingLLM"""
    try:
        print("Testing AnythingLLM connection...")
        print(f"API Key: {ANYTHINGLLM_API_KEY[:10]}...")
        print(f"Base URL: {ANYTHINGLLM_BASE_URL}")
        print(f"Workspace ID: {ANYTHINGLLM_WORKSPACE_ID}")
        
        # Try different possible endpoints
        endpoints_to_try = [
            "/api/workspace/chat",
            "/api/workspace/stream-chat", 
            "/api/chat",
            "/api/v1/chat",
            "/api/workspace/chat-stream",
            "/api/workspace/stream"
        ]
        
        async with httpx.AsyncClient() as client:
            for endpoint in endpoints_to_try:
                try:
                    print(f"\nTrying endpoint: {endpoint}")
                    response = await client.post(
                        f"{ANYTHINGLLM_BASE_URL}{endpoint}",
                        headers={
                            "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "message": "Hello! Please respond with 'Connection successful' if you can read this.",
                            "workspaceId": ANYTHINGLLM_WORKSPACE_ID,
                            "mode": "chat"
                        },
                        timeout=30.0
                    )
                    
                    print(f"Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Connection successful with {endpoint}!")
                        print(f"Response: {result}")
                        return endpoint
                    elif response.status_code == 404:
                        print(f"❌ Endpoint not found: {endpoint}")
                    else:
                        print(f"❌ Error with {endpoint}: {response.text}")
                        
                except Exception as e:
                    print(f"❌ Error with {endpoint}: {str(e)}")
                    continue
            
            # If no endpoint worked, try a simple GET request to check if server is running
            print(f"\nTrying basic connectivity test...")
            try:
                response = await client.get(f"{ANYTHINGLLM_BASE_URL}/", timeout=10.0)
                print(f"Server response: {response.status_code}")
                if response.status_code in [200, 404]:  # 404 is ok, means server is running
                    print("✅ Server is running, but API endpoints might be different")
                    return None
            except Exception as e:
                print(f"❌ Server not reachable: {str(e)}")
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
            }
        ]
        
        user_input = "casual outfit for brunch"
        
        prompt = f"""
You are a fashion AI assistant. Based on the following wardrobe data, suggest an outfit for: "{user_input}"

Wardrobe Data:
{json.dumps(wardrobe_data, indent=2)}

Please analyze the wardrobe and suggest 2-4 items that would make a good outfit for the occasion. 
Return ONLY a JSON array of image paths (no other text, no explanations).

Example format: ["uploads/test1.jpg", "uploads/test2.jpg"]
"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ANYTHINGLLM_BASE_URL}/api/chat",
                headers={
                    "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "message": prompt,
                    "workspaceId": ANYTHINGLLM_WORKSPACE_ID,
                    "mode": "chat"
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Outfit generation test successful!")
                print(f"LLM Response: {result.get('textResponse', 'No response')}")
                return True
            else:
                print(f"❌ Outfit generation failed!")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Outfit generation error: {str(e)}")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 Testing AnythingLLM Integration")
        print("=" * 50)
        
        # Test basic connection
        connection_ok = await test_anythingllm_connection()
        
        if connection_ok:
            # Test outfit generation
            await test_outfit_generation()
        
        print("\n" + "=" * 50)
        if connection_ok:
            print("🎉 AnythingLLM is ready to use!")
        else:
            print("⚠️  Please check your AnythingLLM setup:")
            print("   1. Make sure AnythingLLM is running on http://localhost:3001")
            print("   2. Verify your API key is correct")
            print("   3. Check if the workspace ID is correct")
    
    asyncio.run(main())
