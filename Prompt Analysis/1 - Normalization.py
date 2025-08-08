import pandas as pd

# Step 1: Load the Excel file
file_path = 'RA_analysis.xlsx'
df = pd.read_excel(file_path, sheet_name='original', engine='openpyxl')

# Optional: Inspect the structure
print("Column Names and Data Types:")
print(df.dtypes)
print("\nPreview of First Few Rows:")
print(df.head())

# Step 2: Group and merge data by 'analysis_ID'
# Concatenate 'original' texts and aggregate labels using logical OR
grouped_df = df.groupby('analysis_ID').agg({
    'original': lambda texts: ' '.join(texts),
    'func_bizstr': 'max',
    'edu_stem': 'max',
    'entre_training_experience': 'max'
}).reset_index()

# Save the grouped data to a new Excel file
grouped_df.to_excel('grouped_RA_analysis.xlsx', index=False)

# Optional: Preview the grouped data
print("\nGrouped Data Preview:")
print(grouped_df.head())
