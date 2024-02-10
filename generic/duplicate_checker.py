import pandas as pd

# Path to your CSV file
file_path = 'generic-db.csv'  # Replace with your actual file path

# Read the CSV file
df = pd.read_csv(file_path)

# Define the columns to check for duplicates
columns_to_check = ['day', 'start_time', 'end_time', 'hours', 'classroom']

# Find duplicates
duplicates = df[df.duplicated(subset=columns_to_check, keep=False)]

# Sort by day and start_time
duplicates.sort_values(by=['day', 'start_time'], inplace=True)

# Check if there are any duplicates and print results
if not duplicates.empty:
    print("Duplicate rows based on columns:", columns_to_check)
    print(duplicates)
    print(f"Number of duplicate rows: {len(duplicates)}")
else:
    print("No duplicates found based on the specified columns.")