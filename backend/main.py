
import os
import json
import sqlite3
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

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

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
	processed = 0
	conn = sqlite3.connect("fashion.db")
	c = conn.cursor()
	for file in files:
		# Save image
		image_path = os.path.join("uploads", file.filename)
		with open(image_path, "wb") as f:
			content = await file.read()
			f.write(content)

		# Dummy embedding (random numpy array)
		embedding = np.random.rand(128).astype(np.float32)
		embedding_bytes = embedding.tobytes()

		# Dummy tags
		tags = {"color": "red", "category": "shirt", "style": "casual"}
		tags_json = json.dumps(tags)

		# Store in DB
		c.execute(
			"INSERT INTO clothes (image_path, tags, embedding) VALUES (?, ?, ?)",
			(image_path, tags_json, embedding_bytes)
		)
		processed += 1
	conn.commit()
	conn.close()
	return JSONResponse({"success": True, "images_processed": processed})

# Dummy LLM function (placeholder)
def call_llm(user_input):
	# Placeholder: pretend LLM returns tags to search for
	# In reality, you would call your LLM here
	return {"color": "red", "category": "shirt"}

@app.post("/search")
async def search_images(user_input: str):
	# Call LLM (placeholder)
	tags = call_llm(user_input)
	# Search DB for matching images
	conn = sqlite3.connect("fashion.db")
	c = conn.cursor()
	# Simple search: match color and category in tags JSON
	query = "SELECT id, image_path, tags FROM clothes"
	c.execute(query)
	results = []
	for row in c.fetchall():
		db_tags = json.loads(row[2])
		if all(tags.get(k) == db_tags.get(k) for k in tags):
			results.append({"id": row[0], "image_path": row[1], "tags": db_tags})
	conn.close()
	return {"selected_images": results}

# Install instructions
"""
Install requirements:
	pip install -r requirements.txt

To run the app:
	uvicorn main:app --reload
"""