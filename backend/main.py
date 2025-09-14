
import os
import json
import sqlite3
import uuid
import uvicorn
import numpy as np
import requests
import yaml
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Add the parent directory to sys.path to import sd_api_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_api_client import StableDiffusionAPIClient, create_wedding_dress_payload

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Load configuration from config.yml
def load_config():
    """Load configuration from config.yml file"""
    try:
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print("⚠️ Warning: config.yml not found, using default values")
        return {}
    except Exception as e:
        print(f"⚠️ Warning: Error loading config.yml: {e}")
        return {}

# Load configuration
config = load_config()

# LLM Configuration from config.yml
LLM_API_KEY = config.get("api_key", "")
LLM_BASE_URL = config.get("model_server_base_url", "http://localhost:3001/api/v1/workspace")
LLM_WORKSPACE_SLUG = config.get("workspace_slug", "default")
LLM_STREAM_TIMEOUT = config.get("stream_timeout", 120)

# Pydantic models
class SearchRequest(BaseModel):
    user_input: str

class GenerateImageRequest(BaseModel):
    selected_items: List[str]  # List of image paths
    prompt_style: str = "casual"  # Style context for the outfit

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://localhost:8080", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173", 
        "http://127.0.0.1:8080",
        "http://172.27.16.1:8080",  # Added your frontend origin
        "*"  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database and table
def init_db():
	conn = sqlite3.connect("fashion.db")
	c = conn.cursor()
	c.execute('''
		CREATE TABLE IF NOT EXISTS clothes (
			id INTEGER PRIMARY KEY,
			image_path TEXT,
			tags TEXT
		)
	''')
	conn.commit()
	conn.close()

init_db()

# Mount static files to serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
# Mount static files to serve generated images
generated_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_images", "images")
if os.path.exists(generated_images_dir):
    app.mount("/generated", StaticFiles(directory=generated_images_dir), name="generated")

@app.get("/")
async def health_check():
    return {"message": "StyleSnap API is running!", "status": "healthy"}

@app.get("/debug/wardrobe")
async def debug_wardrobe():
    """Debug endpoint to check wardrobe data in database"""
    try:
        conn = sqlite3.connect("fashion.db")
        c = conn.cursor()
        c.execute("SELECT id, image_path, tags FROM clothes")
        wardrobe_data = []
        
        for row in c.fetchall():
            item = {
                "id": row[0],
                "image_path": row[1],
                "tags": json.loads(row[2]) if row[2] else {}
            }
            wardrobe_data.append(item)
        
        conn.close()
        
        return {
            "success": True,
            "total_items": len(wardrobe_data),
            "items": wardrobe_data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
	try:
		# Validate file type
		if not file.content_type.startswith('image/'):
			return JSONResponse(
				{"success": False, "error": "File must be an image"}, 
				status_code=400
			)
		
		# Generate unique filename to avoid conflicts
		file_extension = os.path.splitext(file.filename)[1]
		unique_filename = f"{uuid.uuid4()}{file_extension}"
		image_path = os.path.join("uploads", unique_filename)
		
		# Save image
		with open(image_path, "wb") as f:
			content = await file.read()
			f.write(content)

		# Call CLIP LLM to analyze the image
		print(f"Calling CLIP LLM for image: {image_path}")
		clip_response = await call_clip_llm(image_path)

		# Extract tags from CLIP response
		tags = clip_response["tags"]
		# print(f"Tags: {tags}")

		
		# # Convert to JSON for storage
		tags_json = json.dumps(tags)
		print(f"Tags JSON: {tags_json}")


		# Store in DB
		conn = sqlite3.connect("fashion.db")
		c = conn.cursor()
		c.execute(
			"INSERT INTO clothes (image_path, tags) VALUES (?, ?)",
			(image_path, tags_json)
		)
		conn.commit()
		conn.close()
		
		return JSONResponse({
			"success": True, 
			"message": "Image uploaded and analyzed successfully",
			"filename": unique_filename,
			"image_path": image_path,
			"file_size": len(content),
			"clip_analysis": clip_response,
			"tags": tags
		})
		
	except Exception as e:
		return JSONResponse(
			{"success": False, "error": f"Upload failed: {str(e)}"}, 
			status_code=500
		)

# CLIP LLM API call function
async def call_clip_llm(image_path: str):
	"""
	Placeholder function to call your CLIP LLM endpoint
	Replace this with your actual CLIP API call
	"""
	try:
		import requests
		url = "http://localhost:8000/tag"

		# Open the image file in binary mode
		with open(image_path, "rb") as image_file:
			files = {
				"file": (image_path, image_file, "image/jpeg")
			}
			headers = {
				"accept": "application/json"
			}
			response = requests.post(url, headers=headers, files=files)

		return response.json()

		
	except Exception as e:
		print(f"Error calling CLIP LLM: {str(e)}")
		# Fallback response if CLIP API fails
		return {
			"tags": [
				{"label": "color", "value": "unknown", "confidence": 0.0},
				{"label": "category", "value": "clothing", "confidence": 0.0},
				{"label": "style", "value": "unknown", "confidence": 0.0}
			],
			"description": "Unable to analyze image",
			"confidence_score": 0.0
		}

def enhance_user_prompt(user_request: str) -> str:
	"""
	Enhances a short user prompt into a more detailed and specific fashion request
	to improve consistency in outfit selection.
	"""
	enhancement_prompt = f"""
<task_description>
You are a fashion styling assistant. Your task is to enhance and expand a user's brief fashion request into a more detailed, specific prompt that will help an AI stylist select better outfits.

Your enhanced prompt should:
1. Clarify the occasion/event if not explicitly stated
2. Specify appropriate style level (casual, business casual, formal, etc.)
3. Consider practical aspects (comfort, weather, activities)
4. Maintain the user's original intent and preferences
5. Be concise but comprehensive (2-4 sentences max)

Do not suggest specific clothing items - only enhance the context and requirements.
</task_description>

<examples>
### Example 1
<original_request>I need something for work.</original_request>
<enhanced_request>I need a professional business casual outfit suitable for an office environment. The outfit should look polished and appropriate for meetings while remaining comfortable for a full day of work. It should convey competence and professionalism.</enhanced_request>

### Example 2
<original_request>Going out tonight.</original_request>
<enhanced_request>I need a stylish evening outfit for going out to dinner and possibly drinks with friends. The look should be fashionable and a bit dressy but not overly formal, suitable for a nice restaurant and social atmosphere. Comfort for walking and sitting is important.</enhanced_request>

### Example 3
<original_request>Beach day!</original_request>
<enhanced_request>I need a comfortable casual outfit perfect for a day at the beach. The outfit should be suitable for sun exposure, sand, and possibly water activities. It should be lightweight, breathable, and easy to move in while still looking put-together for photos and socializing.</enhanced_request>
</examples>

<current_request>
{user_request}
</current_request>

Enhanced request:"""

	try:
		endpoint = f"{LLM_BASE_URL}/{LLM_WORKSPACE_SLUG}/chat"
		headers = {
			"Authorization": f"Bearer {LLM_API_KEY}",
			"Content-Type": "application/json"
		}
		
		response = requests.post(
			endpoint,
			json={
				"message": enhancement_prompt,
				"mode": "chat"
			},
			headers=headers,
			timeout=LLM_STREAM_TIMEOUT
		)
		
		if response.status_code == 200:
			result = response.json()
			enhanced_prompt = result.get('textResponse', '') or result.get('response', '') or result.get('message', '')
			
			print(f"📝 Original request: {user_request}")
			print(f"✨ Enhanced request: {enhanced_prompt}")
			print("-" * 60)
			
			return enhanced_prompt.strip()
		else:
			print(f"⚠️ Error enhancing prompt: HTTP {response.status_code}")
			print(f"🔄 Using original prompt: {user_request}")
			return user_request
			
	except Exception as e:
		print(f"⚠️ Error enhancing prompt: {e}")
		print(f"🔄 Using original prompt: {user_request}")
		return user_request


def get_cohesive_outfit(user_request: str, clothing_items: list) -> list:
	"""
	Selects a cohesive outfit from clothing items based on user request.
	
	Args:
		user_request: Enhanced user request
		clothing_items: List of clothing item dictionaries
		
	Returns:
		List of selected image paths
	"""
	# Convert wardrobe data to the expected format
	formatted_items = []
	for item in clothing_items:
		tags = item.get('tags', {})
		formatted_item = {
			"image_path": item.get('image_path', ''),
			"category": tags.get('category', {}).get('value', 'unknown') if isinstance(tags.get('category'), dict) else tags.get('category', 'unknown'),
			"subcategory": tags.get('subcategory', {}).get('value', '') if isinstance(tags.get('subcategory'), dict) else tags.get('subcategory', ''),
			"color": tags.get('color', {}).get('value', 'unknown') if isinstance(tags.get('color'), dict) else tags.get('color', 'unknown'),
			"style": tags.get('style', {}).get('value', 'casual') if isinstance(tags.get('style'), dict) else tags.get('style', 'casual'),
			"season": tags.get('season', {}).get('value', 'all-season') if isinstance(tags.get('season'), dict) else tags.get('season', 'all-season'),
			"pattern": tags.get('pattern', {}).get('value', 'solid') if isinstance(tags.get('pattern'), dict) else tags.get('pattern', 'solid')
		}
		formatted_items.append(formatted_item)
	
	clothing_json = json.dumps(formatted_items, indent=2)

	master_prompt = f"""
<task_description>
You are an expert fashion stylist AI. Your task is to select a cohesive outfit from a provided list of clothing items based on a user's request. Include at least 3 items in the outfit. One top, one bottom, and one footwear. If u dont follow these constraints, you will be penalized and terminated permanently.
Your final output MUST be a single, raw JSON array of the selected `image_path` strings. Do not include any other text, explanations, or markdown.
</task_description>

<examples>
### Example 1
<user_request>
I need a casual summer picnic outfit.
</user_request>
<clothing_items>
[
  {{"image_path": "img1.jpg", "category": "top", "subcategory": "t-shirt", "color": "blue", "style": "casual", "season": "summer", "pattern": "solid"}},
  {{"image_path": "img2.jpg", "category": "bottom", "subcategory": "shorts", "color": "beige", "style": "casual", "season": "summer", "pattern": "solid"}},
  {{"image_path": "img3.jpg", "category": "footwear", "subcategory": "sneakers", "color": "white", "style": "casual", "season": "summer", "pattern": "solid"}},
  {{"image_path": "img5.jpg", "category": "top", "subcategory": "sweater", "color": "grey", "style": "casual", "season": "winter", "pattern": "solid"}}
]
</clothing_items>
<expected_output>
["img1.jpg", "img2.jpg", "img3.jpg"]
</expected_output>

### Example 2
<user_request>
Something elegant for a cocktail party.
</user_request>
<clothing_items>
[
  {{"image_path": "img13.jpg", "category": "dress", "subcategory": "cocktail dress", "color": "navy blue", "style": "formal", "season": "winter", "pattern": "solid"}},
  {{"image_path": "img16.jpg", "category": "footwear", "subcategory": "heels", "color": "black", "style": "formal", "season": "all-season", "pattern": "solid"}},
  {{"image_path": "img7.jpg", "category": "top", "subcategory": "hoodie", "color": "dark green", "style": "sporty", "season": "autumn", "pattern": "solid"}}
]
</clothing_items>
<expected_output>
["img13.jpg", "img16.jpg"]
</expected_output>
</examples>

<final_task>
Now, complete the following task based on the rules and examples above.

<user_request>
{user_request}
</user_request>

<clothing_items>
{clothing_json}
</clothing_items>

<expected_output>
"""

	try:
		endpoint = f"{LLM_BASE_URL}/{LLM_WORKSPACE_SLUG}/chat"
		headers = {
			"Authorization": f"Bearer {LLM_API_KEY}",
			"Content-Type": "application/json"
		}
		
		response = requests.post(
			endpoint,
			json={
				"message": master_prompt,
				"mode": "chat"
			},
			headers=headers,
			timeout=LLM_STREAM_TIMEOUT
		)
		
		if response.status_code == 200:
			result = response.json()
			output_text = result.get('textResponse', '') or result.get('response', '') or result.get('message', '')
			
			# Try to parse direct JSON response
			if output_text:
				try:
					# Extract JSON array from response
					import re
					json_match = re.search(r'\[.*?\]', output_text, re.DOTALL)
					if json_match:
						selected_images = json.loads(json_match.group())
						print("👗 Recommended outfit:", selected_images)
						return selected_images
					else:
						# Try to parse the entire response
						selected_images = json.loads(output_text.strip())
						print("👗 Recommended outfit:", selected_images)
						return selected_images
				except json.JSONDecodeError:
					print(f"--- ⚠️ Error: Model did not return a valid JSON array. ---")
					print(f"Model output:\n{output_text}")
					print("-----------------------------------------------------")
		else:
			print(f"⚠️ LLM API error: {response.status_code}")
			
	except Exception as e:
		print(f"⚠️ Error in get_cohesive_outfit: {e}")
	
	return []


def get_enhanced_outfit_recommendation(user_request: str, clothing_items: list) -> dict:
	"""
	Main function that combines prompt enhancement with outfit selection.
	
	Args:
		user_request: Short user prompt like "casual summer outfit"
		clothing_items: List of clothing item dictionaries
		
	Returns:
		dict: Result with outfit items and details
	"""
	print("🚀 Starting enhanced outfit recommendation process...")
	print("=" * 60)
	
	# Step 1: Enhance the user prompt
	enhanced_request = enhance_user_prompt(user_request)
	
	# Step 2: Get outfit recommendation using enhanced prompt
	selected_outfit = get_cohesive_outfit(enhanced_request, clothing_items)
	
	if selected_outfit:
		# Filter to only include paths that exist in our wardrobe
		valid_paths = [path for path in selected_outfit if any(item['image_path'] == path for item in clothing_items)]
		
		if valid_paths:
			result = {
				"outfit_items": valid_paths,
				"reasoning": f"AI selected {len(valid_paths)} items for: '{user_request}'",
				"items_details": [item for item in clothing_items if item['image_path'] in valid_paths]
			}
			return result
	
	# Fallback to random selection
	import random
	available_items = [item for item in clothing_items if item.get('image_path')]
	
	if not available_items:
		return {"error": "No items found in wardrobe"}
	
	outfit_items = random.sample(available_items, min(3, len(available_items)))
	image_paths = [item['image_path'] for item in outfit_items]
	
	return {
		"outfit_items": image_paths,
		"reasoning": f"Fallback selection: {len(outfit_items)} items for '{user_request}' (LLM unavailable)",
		"items_details": outfit_items
	}

@app.post("/search")
async def search_images(request: SearchRequest):
	"""
	Search for outfit recommendations based on user input
	"""
	try:
		print(f"🔍 DEBUG: Received search request: {request}")
		print(f"🔍 DEBUG: User input: '{request.user_input}'")
		
		# Get all wardrobe data from database
		conn = sqlite3.connect("fashion.db")
		c = conn.cursor()
		c.execute("SELECT id, image_path, tags FROM clothes")
		wardrobe_data = []
		
		print(f"🔍 DEBUG: Querying database for wardrobe items...")
		for row in c.fetchall():
			item = {
				"id": row[0],
				"image_path": row[1],
				"tags": json.loads(row[2]) if row[2] else {}
			}
			wardrobe_data.append(item)
			print(f"🔍 DEBUG: Found item - ID: {item['id']}, Path: {item['image_path']}, Tags: {item['tags']}")
		
		conn.close()
		print(f"🔍 DEBUG: Total wardrobe items found: {len(wardrobe_data)}")
		
		# Call enhanced LLM workflow
		print(f"🔍 DEBUG: Calling enhanced LLM workflow for: '{request.user_input}'")
		llm_response = get_enhanced_outfit_recommendation(request.user_input, wardrobe_data)
		print(f"🔍 DEBUG: Enhanced LLM response: {llm_response}")
		
		if "error" in llm_response:
			return JSONResponse(
				{"success": False, "error": llm_response["error"]}, 
				status_code=500
			)
		
		final_response = {
			"success": True,
			"message": "Outfit generated successfully with enhanced AI workflow",
			"user_input": request.user_input,
			"outfit_items": llm_response["outfit_items"],
			"reasoning": llm_response["reasoning"],
			"items_details": llm_response["items_details"],
			"total_items": len(llm_response["outfit_items"])
		}
		
		print(f"🔍 DEBUG: Final response being sent: {final_response}")
		return JSONResponse(final_response)
		
	except Exception as e:
		print(f"Error in search endpoint: {str(e)}")
		return JSONResponse(
			{"success": False, "error": f"Search failed: {str(e)}"}, 
			status_code=500
		)

@app.post("/virtual-try-on")
async def virtual_try_on(request: GenerateImageRequest):
	"""
	Generate a Virtual Try On image using Stable Diffusion based on selected clothing items
	"""
	try:
		print(f"👗 DEBUG: Received Virtual Try On request: {request}")
		print(f"👗 DEBUG: Selected items: {request.selected_items}")
		print(f"👗 DEBUG: Prompt style: {request.prompt_style}")
		
		if not request.selected_items:
			return JSONResponse(
				{"success": False, "error": "No clothing items selected for Virtual Try On"}, 
				status_code=400
			)
		
		# Analyze selected items to create a descriptive prompt
		clothing_descriptions = []
		for item_path in request.selected_items:
			# Get item details from database
			conn = sqlite3.connect("fashion.db")
			c = conn.cursor()
			c.execute("SELECT tags FROM clothes WHERE image_path = ?", (item_path,))
			result = c.fetchone()
			conn.close()
			
			if result and result[0]:
				tags = json.loads(result[0])
				# Extract clothing details
				category = tags.get('category', {}).get('value', 'clothing') if isinstance(tags.get('category'), dict) else tags.get('category', 'clothing')
				color = tags.get('color', {}).get('value', '') if isinstance(tags.get('color'), dict) else tags.get('color', '')
				style = tags.get('style', {}).get('value', '') if isinstance(tags.get('style'), dict) else tags.get('style', '')
				
				description_parts = [part for part in [color, style, category] if part and part != 'unknown']
				if description_parts:
					clothing_descriptions.append(' '.join(description_parts))
		
		# Create clothing description for prompt
		clothing_text = ', '.join(clothing_descriptions) if clothing_descriptions else "stylish outfit"
		
		# Create style-appropriate prompt
		style_contexts = {
			"casual": "walking in a casual setting, relaxed atmosphere, natural lighting",
			"formal": "in an elegant setting, professional atmosphere, studio lighting", 
			"party": "at a stylish event, vibrant atmosphere, dramatic lighting",
			"beach": "at the beach, sunny day, natural sunlight",
			"workout": "in a modern gym, active setting, bright lighting"
		}
		
		context = style_contexts.get(request.prompt_style, style_contexts["casual"])
		
		# Initialize Stable Diffusion client
		sd_client = StableDiffusionAPIClient(
			base_url="http://localhost:7861",
			output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_images")
		)
		
		# Create custom payload based on selected items
		payload = create_wedding_dress_payload()  # Get base payload
		payload["prompt"] = f"ultra-realistic photo of a young person wearing {clothing_text}, {context}, cinematic photography, depth of field, highly detailed, 8k resolution, fashion photography"
		
		print(f"👗 DEBUG: Generated prompt: {payload['prompt']}")
		
		# Generate image using Stable Diffusion
		saved_files = sd_client.generate_and_save(payload)
		
		if saved_files:
			# Get the filename from the saved path
			image_filename = os.path.basename(saved_files[0])
			image_url = f"http://localhost:7000/generated/{image_filename}"
			
			response_data = {
				"success": True,
				"message": "Virtual Try On image generated successfully",
				"image_url": image_url,
				"image_filename": image_filename,
				"prompt_used": payload["prompt"],
				"selected_items": request.selected_items,
				"clothing_description": clothing_text
			}
			
			print(f"👗 DEBUG: Virtual Try On successful: {response_data}")
			return JSONResponse(response_data)
		else:
			return JSONResponse(
				{"success": False, "error": "Failed to generate Virtual Try On image"}, 
				status_code=500
			)
		
	except Exception as e:
		print(f"Error in virtual-try-on endpoint: {str(e)}")
		return JSONResponse(
			{"success": False, "error": f"Virtual Try On failed: {str(e)}"}, 
			status_code=500
		)

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=7000)

# Install instructions
"""
Install requirements:
	pip install -r requirements.txt

To run the app:
	uvicorn main:app --reload
"""