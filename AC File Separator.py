import pandas as pd
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def split_and_save(file_path):
    p = Path(file_path)
    print(f"Processing: {p.name}")

    with open(p, 'r') as f:
        lines = f.readlines()
    
    header_idx = next(i for i, line in enumerate(lines) if 'AC Level' in line)
    metadata = lines[:header_idx]

    df = pd.read_csv(p, skiprows=header_idx, index_col=False)
    df.columns = df.columns.str.strip()

    if 'AC Level (V)' not in df.columns:
        print("Error: 'AC Level (V)' column not found.")
        return

    unique_levels = df['AC Level (V)'].unique()
    print(f"Found {len(unique_levels)} AC Levels: {unique_levels}")

    # Enumerate provides a counter (i) starting at 1
    for i, ac_val in enumerate(unique_levels, start=1):
        subset = df[df['AC Level (V)'] == ac_val]
        
        # Format: Sequence_ACLevel.csv (e.g., 1_0.005.csv)
        output_name = f"{i}_{ac_val:g}.csv"
        output_path = p.parent / output_name
        
        print(f"Saving {len(subset)} rows to {output_name}...")
        
        with open(output_path, 'w') as f_out:
            f_out.writelines(metadata)
            subset.to_csv(f_out, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Select CSV file(s)",
        filetypes=[("CSV Files", "*.csv")]
    )
    
    for path in file_paths:
        split_and_save(path)
        
    print("Done.")