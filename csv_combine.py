import os
import glob
import pandas as pd
import logging_function
import logging
logger = logging.getLogger(__name__)

# Directory containing the CSV files
csv_dir = '.'  # Change this to your directory if needed

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# List to hold DataFrames
dfs = []

for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    # Remove the 'source field' column if it exists
    if 'TargetField' in df.columns:
        df = df.drop(columns=['TargetField'])
    # Add a new column 'table' with the filename (without extension)
    table_name = os.path.splitext(os.path.basename(file))[0]
    df['table'] = table_name
    dfs.append(df)

# Concatenate all DataFrames
master_df = pd.concat(dfs, ignore_index=True)

# Write to master_list.csv
master_df.to_csv('master_list.csv', index=False)

print("Combined CSV written to master_list.csv")
