import duckdb
import glob
import os
import pandas as pd

print("🚀 Creating full NH database with ALL quarters...")

# Create database
conn = duckdb.connect('nh_staffing_full.db')

# Drop existing table if it exists
conn.execute("DROP TABLE IF EXISTS non_nurse_staffing_full")

# Load all CSV files
csv_files = glob.glob('standardized_NonNurse/PBJ_dailynonnursestaffing_*.csv')
csv_files.sort()

print(f"Found {len(csv_files)} CSV files")

total_records = 0

# Process each file using pandas first
for i, file in enumerate(csv_files):
    quarter = file.split('_')[-1].replace('.csv', '')
    print(f"Processing {quarter}...")
    
    try:
        # Read with pandas to ensure all rows are loaded
        df = pd.read_csv(file, low_memory=False)
        print(f"  📊 Read {len(df):,} rows from {quarter}")
        
        if i == 0:
            # First file - create table from it
            conn.execute("CREATE TABLE non_nurse_staffing_full AS SELECT * FROM df")
            print(f"  ✅ Created table from {quarter}")
        else:
            # Subsequent files - append to table
            conn.execute("INSERT INTO non_nurse_staffing_full SELECT * FROM df")
            print(f"  ✅ Appended {quarter}")
        
        total_records += len(df)
        print(f"  📈 Total records so far: {total_records:,}")
        
    except Exception as e:
        print(f"  ❌ Error with {quarter}: {e}")
        continue

# Create indexes
print("Creating indexes...")
conn.execute("CREATE INDEX IF NOT EXISTS idx_provnum ON non_nurse_staffing_full(PROVNUM)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_provnum_date ON non_nurse_staffing_full(PROVNUM, WorkDate)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON non_nurse_staffing_full(WorkDate)")

# Create NH lookup table
print("Creating NH lookup table...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS nh_lookup AS
    SELECT DISTINCT 
        PROVNUM,
        PROVNAME,
        CITY,
        STATE,
        COUNTY_NAME,
        COUNT(*) as total_records,
        MIN(WorkDate) as first_date,
        MAX(WorkDate) as last_date
    FROM non_nurse_staffing_full
    GROUP BY PROVNUM, PROVNAME, CITY, STATE, COUNTY_NAME
    ORDER BY PROVNAME
""")

conn.execute("CREATE INDEX IF NOT EXISTS idx_lookup_provnum ON nh_lookup(PROVNUM)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_lookup_name ON nh_lookup(PROVNAME)")

# Get final stats
stats = conn.execute("SELECT COUNT(*) FROM non_nurse_staffing_full").fetchone()
nh_count = conn.execute("SELECT COUNT(*) FROM nh_lookup").fetchone()

print(f"\n🎉 DATABASE CREATED SUCCESSFULLY!")
print(f"📊 Total records: {stats[0]:,}")
print(f"🏥 Total nursing homes: {nh_count[0]:,}")
print(f"💾 Database size: {os.path.getsize('nh_staffing_full.db') / (1024*1024):.1f} MB")

conn.close()
print("✅ Done! Database ready for fast queries.") 