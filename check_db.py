import sqlite3
import os

def check_db():
    try:
        db_path = 'pbj.db'
        print(f"Database exists: {os.path.exists(db_path)}")
        print(f"Database size: {os.path.getsize(db_path)} bytes")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("\nTables in database:")
        for table in tables:
            print(f"- {table[0]}")
            
            # Get schema for each table
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            print("  Columns:")
            for col in columns:
                print(f"    {col[1]} ({col[2]})")
            print()
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_db() 