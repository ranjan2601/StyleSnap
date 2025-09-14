#!/usr/bin/env python3
"""
Script to populate the database with dummy clothing data for testing
"""
import sqlite3
import json
import numpy as np
import uuid
import os

def populate_dummy_data():
    """Add dummy clothing items to the database"""
    
    # Connect to database
    conn = sqlite3.connect("fashion.db")
    c = conn.cursor()
    
    # Clear existing data
    c.execute("DELETE FROM clothes")
    
    # Create dummy clothing items
    dummy_items = [
        {
            "image_path": "uploads/party_dress_1.jpg",
            "tags": {
                "color": {"value": "red", "confidence": 0.95},
                "category": {"value": "dress", "confidence": 0.98},
                "style": {"value": "party", "confidence": 0.92},
                "material": {"value": "silk", "confidence": 0.88},
                "season": {"value": "all", "confidence": 0.85},
                "occasion": {"value": "party", "confidence": 0.96},
                "description": "Elegant red party dress",
                "confidence_score": 0.92,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "red", "confidence": 0.95},
                        {"label": "category", "value": "dress", "confidence": 0.98},
                        {"label": "style", "value": "party", "confidence": 0.92}
                    ],
                    "description": "Elegant red party dress",
                    "confidence_score": 0.92
                }
            }
        },
        {
            "image_path": "uploads/party_dress_2.jpg",
            "tags": {
                "color": {"value": "black", "confidence": 0.97},
                "category": {"value": "dress", "confidence": 0.99},
                "style": {"value": "elegant", "confidence": 0.94},
                "material": {"value": "sequin", "confidence": 0.91},
                "season": {"value": "winter", "confidence": 0.89},
                "occasion": {"value": "party", "confidence": 0.98},
                "description": "Black sequin party dress",
                "confidence_score": 0.94,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "black", "confidence": 0.97},
                        {"label": "category", "value": "dress", "confidence": 0.99},
                        {"label": "style", "value": "elegant", "confidence": 0.94}
                    ],
                    "description": "Black sequin party dress",
                    "confidence_score": 0.94
                }
            }
        },
        {
            "image_path": "uploads/casual_shirt.jpg",
            "tags": {
                "color": {"value": "blue", "confidence": 0.93},
                "category": {"value": "shirt", "confidence": 0.96},
                "style": {"value": "casual", "confidence": 0.89},
                "material": {"value": "cotton", "confidence": 0.87},
                "season": {"value": "summer", "confidence": 0.91},
                "occasion": {"value": "casual", "confidence": 0.94},
                "description": "Blue casual cotton shirt",
                "confidence_score": 0.90,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "blue", "confidence": 0.93},
                        {"label": "category", "value": "shirt", "confidence": 0.96},
                        {"label": "style", "value": "casual", "confidence": 0.89}
                    ],
                    "description": "Blue casual cotton shirt",
                    "confidence_score": 0.90
                }
            }
        },
        {
            "image_path": "uploads/formal_blazer.jpg",
            "tags": {
                "color": {"value": "navy", "confidence": 0.96},
                "category": {"value": "blazer", "confidence": 0.98},
                "style": {"value": "formal", "confidence": 0.95},
                "material": {"value": "wool", "confidence": 0.92},
                "season": {"value": "all", "confidence": 0.88},
                "occasion": {"value": "business", "confidence": 0.97},
                "description": "Navy formal wool blazer",
                "confidence_score": 0.93,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "navy", "confidence": 0.96},
                        {"label": "category", "value": "blazer", "confidence": 0.98},
                        {"label": "style", "value": "formal", "confidence": 0.95}
                    ],
                    "description": "Navy formal wool blazer",
                    "confidence_score": 0.93
                }
            }
        },
        {
            "image_path": "uploads/party_heels.jpg",
            "tags": {
                "color": {"value": "silver", "confidence": 0.94},
                "category": {"value": "shoes", "confidence": 0.97},
                "style": {"value": "party", "confidence": 0.93},
                "material": {"value": "leather", "confidence": 0.89},
                "season": {"value": "all", "confidence": 0.86},
                "occasion": {"value": "party", "confidence": 0.95},
                "description": "Silver party heels",
                "confidence_score": 0.91,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "silver", "confidence": 0.94},
                        {"label": "category", "value": "shoes", "confidence": 0.97},
                        {"label": "style", "value": "party", "confidence": 0.93}
                    ],
                    "description": "Silver party heels",
                    "confidence_score": 0.91
                }
            }
        },
        {
            "image_path": "uploads/jeans.jpg",
            "tags": {
                "color": {"value": "blue", "confidence": 0.92},
                "category": {"value": "pants", "confidence": 0.95},
                "style": {"value": "casual", "confidence": 0.91},
                "material": {"value": "denim", "confidence": 0.96},
                "season": {"value": "all", "confidence": 0.88},
                "occasion": {"value": "casual", "confidence": 0.93},
                "description": "Blue denim jeans",
                "confidence_score": 0.90,
                "clip_analysis": {
                    "tags": [
                        {"label": "color", "value": "blue", "confidence": 0.92},
                        {"label": "category", "value": "pants", "confidence": 0.95},
                        {"label": "style", "value": "casual", "confidence": 0.91}
                    ],
                    "description": "Blue denim jeans",
                    "confidence_score": 0.90
                }
            }
        }
    ]
    
    # Insert dummy data
    for item in dummy_items:
        # Generate random embedding
        embedding = np.random.rand(128).astype(np.float32)
        embedding_bytes = embedding.tobytes()
        
        # Convert tags to JSON
        tags_json = json.dumps(item["tags"])
        
        # Insert into database
        c.execute(
            "INSERT INTO clothes (image_path, tags, embedding) VALUES (?, ?, ?)",
            (item["image_path"], tags_json, embedding_bytes)
        )
        
        print(f"Added: {item['image_path']} - {item['tags']['description']}")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"\n✅ Successfully added {len(dummy_items)} dummy clothing items to the database!")
    print("\nItems added:")
    for item in dummy_items:
        print(f"  - {item['image_path']}: {item['tags']['description']}")

def create_dummy_images():
    """Create dummy image files for testing"""
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    dummy_images = [
        "party_dress_1.jpg",
        "party_dress_2.jpg", 
        "casual_shirt.jpg",
        "formal_blazer.jpg",
        "party_heels.jpg",
        "jeans.jpg"
    ]
    
    for image_name in dummy_images:
        image_path = os.path.join("uploads", image_name)
        if not os.path.exists(image_path):
            # Create a simple dummy image file (just a placeholder)
            with open(image_path, "wb") as f:
                f.write(b"Dummy image data for " + image_name.encode())
            print(f"Created dummy image: {image_path}")

if __name__ == "__main__":
    print("🚀 Populating database with dummy clothing data...")
    print("=" * 60)
    
    # Create dummy images
    create_dummy_images()
    
    # Populate database
    populate_dummy_data()
    
    print("\n" + "=" * 60)
    print("🎉 Database populated! You can now test with prompts like:")
    print("  - 'party dress'")
    print("  - 'casual outfit'") 
    print("  - 'formal wear'")
    print("  - 'business attire'")
