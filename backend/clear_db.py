#!/usr/bin/env python3
"""
Script to clear all contents from the fashion database
"""

import sqlite3
import os
from datetime import datetime

def clear_database():
    """Clear all records from the clothes table"""
    try:
        # Connect to database
        conn = sqlite3.connect("fashion.db")
        cursor = conn.cursor()
        
        print("🗑️  CLEARING FASHION DATABASE")
        print("=" * 40)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check current count
        cursor.execute("SELECT COUNT(*) FROM clothes")
        current_count = cursor.fetchone()[0]
        print(f"📊 Current items in database: {current_count}")
        
        if current_count == 0:
            print("✅ Database is already empty!")
            conn.close()
            return
        
        # Confirm deletion
        print(f"⚠️  This will DELETE all {current_count} items from the database!")
        print("⚠️  This action cannot be undone!")
        print()
        
        # Delete all records
        cursor.execute("DELETE FROM clothes")
        deleted_count = cursor.rowcount
        
        # Reset auto-increment counter
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='clothes'")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully deleted {deleted_count} items from database")
        print("✅ Auto-increment counter reset")
        print("✅ Database is now empty")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def clear_uploads_folder():
    """Also clear the uploads folder (optional)"""
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            print("📁 Uploads folder doesn't exist")
            return
            
        files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
        
        if not files:
            print("📁 Uploads folder is already empty")
            return
            
        print(f"📁 Found {len(files)} files in uploads folder:")
        for file in files[:5]:  # Show first 5 files
            print(f"   • {file}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more files")
        
        print()
        response = input("🗑️  Do you also want to delete all uploaded image files? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            deleted_files = 0
            for file in files:
                file_path = os.path.join(uploads_dir, file)
                try:
                    os.remove(file_path)
                    deleted_files += 1
                except Exception as e:
                    print(f"❌ Could not delete {file}: {e}")
            
            print(f"✅ Deleted {deleted_files} files from uploads folder")
        else:
            print("📁 Keeping uploaded image files")
            
    except Exception as e:
        print(f"❌ Error clearing uploads: {e}")

def confirm_and_clear():
    """Ask for confirmation before clearing"""
    print("⚠️  WARNING: This will permanently delete ALL clothing items from your database!")
    print("⚠️  Make sure you have a backup if you need to recover this data.")
    print()
    
    response = input("Are you sure you want to clear the database? Type 'YES' to confirm: ").strip()
    
    if response == 'YES':
        clear_database()
        print()
        clear_uploads_folder()
    else:
        print("❌ Operation cancelled - database not cleared")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        # Force clear without confirmation (use with caution!)
        clear_database()
    else:
        confirm_and_clear()
