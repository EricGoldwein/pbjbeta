import sqlite3
import pandas as pd

def show_db_structure():
    conn = sqlite3.connect('pbj_data.db')
    cursor = conn.cursor()
    
    # Get table info
    print("\n=== Database Structure ===")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("\nColumns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get index info
        cursor.execute(f"PRAGMA index_list({table_name});")
        indexes = cursor.fetchall()
        if indexes:
            print("\nIndexes:")
            for idx in indexes:
                idx_name = idx[1]
                cursor.execute(f"PRAGMA index_info({idx_name});")
                idx_cols = cursor.fetchall()
                col_names = [columns[col[1]][1] for col in idx_cols]
                print(f"  {idx_name}: {', '.join(col_names)}")
        
        # Show sample data
        print("\nSample Data (first 2 rows):")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        sample = cursor.fetchall()
        if sample:
            # Get column names
            col_names = [col[1] for col in columns]
            # Create DataFrame for nice display
            df = pd.DataFrame(sample, columns=col_names)
            print(df.to_string())
        
        print("\n" + "="*50)
    
    conn.close()

if __name__ == '__main__':
    show_db_structure() 