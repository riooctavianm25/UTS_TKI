import pandas as pd
import os

# Check all CSV files
for f in os.listdir('.'):
    if f.endswith('.csv'):
        print(f'\n=== File: {f} ===')
        try:
            df = pd.read_csv(f, encoding='latin-1')
            print(f'Shape: {df.shape}')
            print(f'Columns ({len(df.columns)}):')
            for i, col in enumerate(df.columns):
                print(f'  [{i}]: {str(col)[:80]}')
            
            print(f'\nFirst 2 data rows:')
            print(df.head(2).to_string())
        except Exception as e:
            print(f'Error: {e}')
