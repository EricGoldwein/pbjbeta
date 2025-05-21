import pandas as pd
import glob
import os

# Get all quarterly files
quarterly_files = glob.glob('PBJ_dailynursestaffing_CY20*.csv')
quarterly_files.sort()

# Dictionary to store headers for each quarter
headers_dict = {}

# Read headers from each file
for file in quarterly_files:
    quarter = os.path.basename(file).replace('PBJ_dailynursestaffing_', '').replace('.csv', '')
    df = pd.read_csv(file, nrows=0)  # Read only headers
    headers_dict[quarter] = set(df.columns)

# Create output file for comparison
with open('quarterly_headers_comparison.txt', 'w') as f:
    # Write header comparison
    f.write("HEADER COMPARISON ACROSS QUARTERS\n")
    f.write("="*50 + "\n\n")
    
    # Get all unique headers across all files
    all_headers = set()
    for headers in headers_dict.values():
        all_headers.update(headers)
    
    # Create a matrix of header presence
    f.write("Header Presence Matrix:\n")
    f.write("Quarter".ljust(15) + " | " + " | ".join(sorted(all_headers)) + "\n")
    f.write("-" * (15 + 3 + len(" | ".join(sorted(all_headers)))) + "\n")
    
    for quarter, headers in sorted(headers_dict.items()):
        f.write(quarter.ljust(15) + " | ")
        f.write(" | ".join(["X" if header in headers else " " for header in sorted(all_headers)]))
        f.write("\n")
    
    # Check for structural differences
    f.write("\n\nSTRUCTURAL DIFFERENCES\n")
    f.write("="*50 + "\n")
    
    # Compare each quarter with the first quarter
    first_quarter = sorted(headers_dict.keys())[0]
    first_headers = headers_dict[first_quarter]
    
    for quarter, headers in sorted(headers_dict.items()):
        if quarter != first_quarter:
            missing_headers = first_headers - headers
            extra_headers = headers - first_headers
            
            if missing_headers or extra_headers:
                f.write(f"\nDifferences between {first_quarter} and {quarter}:\n")
                if missing_headers:
                    f.write(f"Missing headers in {quarter}: {', '.join(sorted(missing_headers))}\n")
                if extra_headers:
                    f.write(f"Extra headers in {quarter}: {', '.join(sorted(extra_headers))}\n")
    
    # Check for any data type differences in the first row
    f.write("\n\nDATA TYPE COMPARISON (First Row)\n")
    f.write("="*50 + "\n")
    
    for file in quarterly_files:
        quarter = os.path.basename(file).replace('PBJ_dailynursestaffing_', '').replace('.csv', '')
        df = pd.read_csv(file, nrows=1)
        f.write(f"\n{quarter} data types:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"{col}: {dtype}\n")

print("Comparison file created: quarterly_headers_comparison.txt") 