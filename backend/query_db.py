#!/usr/bin/env python3
"""
Simple script to query and display contents of the fashion database
"""

import sqlite3
import json
from datetime import datetime

def query_database():
    """Query and display all contents of the fashion database"""
    try:
        # Connect to database
        conn = sqlite3.connect("fashion.db")
        cursor = conn.cursor()
        
        print("=" * 60)
        print("FASHION DATABASE CONTENTS")
        print("=" * 60)
        print(f"Query Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables in database: {[table[0] for table in tables]}")
        print()
        
        # Get all items from clothes table
        cursor.execute("SELECT id, image_path, tags FROM clothes ORDER BY id")
        items = cursor.fetchall()
        
        if not items:
            print("❌ No items found in the database")
            return
            
        print(f"👕 Total Items: {len(items)}")
        print("-" * 60)
        
        for item in items:
            item_id, image_path, tags_json = item
            
            # Parse tags JSON
            try:
                tags = json.loads(tags_json) if tags_json else {}
            except json.JSONDecodeError:
                tags = {}
            
            print(f"🆔 ID: {item_id}")
            print(f"📁 Image Path: {image_path}")
            
            # Display tags in a readable format
            if tags:
                print("🏷️  Tags:")
                for key, value in tags.items():
                    if isinstance(value, dict) and 'value' in value:
                        confidence = value.get('confidence', 'N/A')
                        print(f"   • {key.capitalize()}: {value['value']} (confidence: {confidence})")
                    elif key == 'description':
                        print(f"   • Description: {value}")
                    elif key == 'confidence_score':
                        print(f"   • Overall Confidence: {value}")
                    else:
                        print(f"   • {key.capitalize()}: {value}")
            else:
                print("🏷️  Tags: None")
            
            print("-" * 40)
        
        conn.close()
        print(f"\n✅ Successfully queried {len(items)} items from database")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def quick_summary():
    """Show a quick summary of database contents"""
    try:
        conn = sqlite3.connect("fashion.db")
        cursor = conn.cursor()
        
        # Count total items
        cursor.execute("SELECT COUNT(*) FROM clothes")
        total_items = cursor.fetchone()[0]
        
        # Get recent items
        cursor.execute("SELECT image_path, tags FROM clothes ORDER BY id DESC LIMIT 5")
        recent_items = cursor.fetchall()
        
        print("📊 QUICK SUMMARY")
        print("=" * 30)
        print(f"Total Items: {total_items}")
        
        if recent_items:
            print("\n🔥 Recent Items:")
            for i, (path, tags_json) in enumerate(recent_items, 1):
                try:
                    tags = json.loads(tags_json) if tags_json else {}
                    description = tags.get('description', 'No description')
                    print(f"  {i}. {path} - {description}")
                except:
                    print(f"  {i}. {path} - Unable to parse tags")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        quick_summary()
    else:
        query_database()
