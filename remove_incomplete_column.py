import pandas as pd

# Read the CSV file
file_path = 'standardized_PBJ/PBJ_dailynursestaffing_CY2021Q4.csv'

# Try different encodings
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"Successfully read file with {encoding} encoding")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    print("Could not read file with any of the attempted encodings")
    exit(1)

# Drop the 'incomplete' column if it exists
if 'incomplete' in df.columns:
    df = df.drop(columns=['incomplete'])
    print("Successfully removed 'incomplete' column")
else:
    print("'incomplete' column not found in the file")

# Save the modified dataframe back to the same file
df.to_csv(file_path, index=False, encoding='utf-8')
print(f"File saved successfully to {file_path}") 