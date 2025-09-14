
import os
import json
import sqlite3
import uuid
import uvicorn
import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Pydantic models
class SearchRequest(BaseModel):
    user_input: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
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
			tags TEXT,
			embedding BLOB
		)
	''')
	conn.commit()
	conn.close()

init_db()

# Mount static files to serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

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
		tags = {}
		for tag in clip_response["tags"]:
			tags[tag["label"]] = {
				"value": tag["value"],
				"confidence": tag["confidence"]
			}
		
		# Add additional metadata
		tags["description"] = clip_response["description"]
		tags["confidence_score"] = clip_response["confidence_score"]
		tags["clip_analysis"] = clip_response
		
		# Convert to JSON for storage
		tags_json = json.dumps(tags)

		# Generate embedding (you can replace this with actual CLIP embeddings)
		embedding = np.random.rand(128).astype(np.float32)
		embedding_bytes = embedding.tobytes()

		# Store in DB
		conn = sqlite3.connect("fashion.db")
		c = conn.cursor()
		c.execute(
			"INSERT INTO clothes (image_path, tags, embedding) VALUES (?, ?, ?)",
			(image_path, tags_json, embedding_bytes)
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
		# TODO: Replace with your actual CLIP LLM endpoint
		# Example implementation:
		# import httpx
		# async with httpx.AsyncClient() as client:
		#     with open(image_path, "rb") as image_file:
		#         response = await client.post(
		#             "http://your-clip-api.com/analyze",
		#             files={"image": image_file},
		#             timeout=30.0
		#         )
		#     if response.status_code == 200:
		#         return response.json()
		#     else:
		#         raise Exception(f"CLIP API error: {response.status_code}")
		
		# Placeholder response - replace with actual API call
		import random
		colors = ["red", "blue", "green", "black", "white", "yellow", "purple", "pink", "orange", "brown"]
		categories = ["shirt", "pants", "dress", "jacket", "shoes", "hat", "bag", "accessory", "sweater", "shorts"]
		styles = ["casual", "formal", "sporty", "vintage", "modern", "bohemian", "minimalist", "elegant", "streetwear", "classic"]
		materials = ["cotton", "denim", "leather", "silk", "wool", "polyester", "linen", "cashmere", "suede", "canvas"]
		
		# Simulate CLIP LLM response
		clip_response = {
			"tags": [
				{"label": "color", "value": random.choice(colors), "confidence": round(random.uniform(0.7, 0.95), 2)},
				{"label": "category", "value": random.choice(categories), "confidence": round(random.uniform(0.8, 0.98), 2)},
				{"label": "style", "value": random.choice(styles), "confidence": round(random.uniform(0.6, 0.9), 2)},
				{"label": "material", "value": random.choice(materials), "confidence": round(random.uniform(0.5, 0.85), 2)},
				{"label": "season", "value": random.choice(["spring", "summer", "fall", "winter"]), "confidence": round(random.uniform(0.4, 0.8), 2)},
				{"label": "occasion", "value": random.choice(["work", "casual", "formal", "party", "sports", "travel"]), "confidence": round(random.uniform(0.5, 0.9), 2)}
			],
			"description": f"A {random.choice(colors)} {random.choice(categories)} in {random.choice(styles)} style",
			"confidence_score": round(random.uniform(0.75, 0.95), 2)
		}
		
		return clip_response
		
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

# Ollama LLM API call function
async def call_ollama_llm(user_input: str, wardrobe_data: list):
	"""
	Call Ollama with Llama 3.2:3b to generate outfit recommendations
	"""
	try:
		print(f"🤖 DEBUG: Starting Ollama LLM call")
		print(f"🤖 DEBUG: User input: '{user_input}'")
		print(f"🤖 DEBUG: Wardrobe data length: {len(wardrobe_data)}")
		
		# Prepare wardrobe data for LLM
		wardrobe_summary = []
		for item in wardrobe_data:
			if item.get('image_path'):
				tags = item.get('tags', {})
				item_info = {
					"image_path": item['image_path'],
					"tags": tags
				}
				wardrobe_summary.append(item_info)
				print(f"🤖 DEBUG: Added to summary - {item['image_path']}: {tags}")
		
		print(f"🤖 DEBUG: Wardrobe summary prepared: {len(wardrobe_summary)} items")
		
		# Create the prompt for Ollama
		prompt = f"""You are a fashion AI assistant. Based on the following wardrobe data, suggest an outfit for: "{user_input}"

Wardrobe Data:
{json.dumps(wardrobe_summary, indent=2)}

Please analyze the wardrobe and suggest 2-4 items that would make a good outfit for the occasion. 
Return ONLY a JSON array of image paths (no other text, no explanations).

Example format: ["uploads/item1.jpg", "uploads/item2.jpg", "uploads/item3.jpg"]"""
		
		print(f"🤖 DEBUG: Prompt created (length: {len(prompt)} chars)")
		print(f"🤖 DEBUG: Prompt preview: {prompt[:200]}...")
		
		# Call Ollama API
		print(f"🤖 DEBUG: Calling Ollama API at {OLLAMA_BASE_URL}/api/generate")
		print(f"🤖 DEBUG: Using model: {OLLAMA_MODEL}")
		
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
			
			print(f"🤖 DEBUG: Ollama API response status: {response.status_code}")
			print(f"🤖 DEBUG: Ollama API response headers: {dict(response.headers)}")
			
			if response.status_code == 200:
				result = response.json()
				llm_response = result.get('response', '')
				
				print(f"🤖 DEBUG: Raw Ollama response: {result}")
				print(f"🤖 DEBUG: Extracted response text: '{llm_response}'")
				
				# Try to parse the JSON response from LLM
				try:
					print(f"🤖 DEBUG: Attempting to parse JSON from LLM response...")
					
					# Extract JSON from the response (in case there's extra text)
					import re
					json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
					if json_match:
						json_str = json_match.group()
						print(f"🤖 DEBUG: Found JSON match: '{json_str}'")
						image_paths = json.loads(json_str)
					else:
						print(f"🤖 DEBUG: No JSON array found, trying to parse entire response...")
						# Fallback: try to parse the entire response
						image_paths = json.loads(llm_response)
					
					print(f"🤖 DEBUG: Parsed image paths: {image_paths}")
					print(f"🤖 DEBUG: Image paths type: {type(image_paths)}")
					
					# Validate that we got image paths
					if isinstance(image_paths, list) and all(isinstance(path, str) for path in image_paths):
						print(f"🤖 DEBUG: Valid JSON array found with {len(image_paths)} paths")
						
						# Filter to only include paths that exist in our wardrobe
						valid_paths = [path for path in image_paths if any(item['image_path'] == path for item in wardrobe_data)]
						print(f"🤖 DEBUG: Valid paths after filtering: {valid_paths}")
						
						if valid_paths:
							result = {
								"outfit_items": valid_paths,
								"reasoning": f"AI selected {len(valid_paths)} items for: '{user_input}'",
								"items_details": [item for item in wardrobe_data if item['image_path'] in valid_paths]
							}
							print(f"🤖 DEBUG: Returning result: {result}")
							return result
						else:
							raise Exception("No valid image paths found in LLM response")
					else:
						raise Exception("Invalid JSON format from LLM")
						
				except json.JSONDecodeError as e:
					print(f"🤖 DEBUG: JSON parsing failed: {str(e)}")
					print(f"🤖 DEBUG: Raw response that failed to parse: '{llm_response}'")
					raise Exception(f"LLM returned invalid JSON: {str(e)}")
			else:
				raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
		
	except Exception as e:
		print(f"Error calling Ollama: {str(e)}")
		# Fallback to simple selection if LLM fails
		import random
		available_items = [item for item in wardrobe_data if item.get('image_path')]
		
		if not available_items:
			return {"error": "No items found in wardrobe"}
		
		# Simple fallback selection
		outfit_items = random.sample(available_items, min(3, len(available_items)))
		image_paths = [item['image_path'] for item in outfit_items]
		
		return {
			"outfit_items": image_paths,
			"reasoning": f"Fallback selection: {len(outfit_items)} items for '{user_input}' (Ollama unavailable)",
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
		
		# Call Ollama LLM with user input and wardrobe data
		print(f"🔍 DEBUG: Calling Ollama LLM for outfit generation: '{request.user_input}'")
		llm_response = await call_ollama_llm(request.user_input, wardrobe_data)
		print(f"🔍 DEBUG: Ollama response: {llm_response}")
		
		if "error" in llm_response:
			return JSONResponse(
				{"success": False, "error": llm_response["error"]}, 
				status_code=500
			)
		
		final_response = {
			"success": True,
			"message": "Outfit generated successfully",
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

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)

# Install instructions
"""
Install requirements:
	pip install -r requirements.txt

To run the app:
	uvicorn main:app --reload
"""