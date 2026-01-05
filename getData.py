import kagglehub
import os
from pathlib import Path
import glob
import pandas as pd
def getRawData():
  # Download latest version
  path = kagglehub.dataset_download("camnugent/california-housing-prices")
  # Folder containing CSV files (set via env var or change here)
  CSV_FOLDER = path
  data_path = Path(CSV_FOLDER)
  csv_files = list(data_path.glob('*.csv')) if data_path.exists() else []
  if csv_files:
      print(f"   Found {len(csv_files)} CSV file(s) in '{data_path}'. Loading...")
      dfs = []
      for p in csv_files:
          try:
              print(f"    - Reading {p}")
              dfs.append(pd.read_csv(p))
          except Exception as e:
              print(f"    ! Failed to read {p}: {e}")
      if dfs:
          # Concatenate, allowing for differing columns
          df = pd.concat(dfs, ignore_index=True, sort=False)
          print(f"   Loaded combined dataframe with shape: {df.shape}")
      else:
          print("   No valid CSVs loaded; falling back to sklearn fetch.")
  else:
      print(f"   No CSV files found in '{data_path}'.")
  return df
