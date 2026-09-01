import os
import json
import pandas as pd
import numpy as np

def audit_and_inspect_dataset(raw_path: str) -> dict:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"10K placement dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    rows, cols = df.shape
    col_names = df.columns.tolist()
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_vals = df.isnull().sum().to_dict()
    dup_rows = int(df.duplicated().sum())
    unique_counts = {col: int(df[col].nunique()) for col in col_names}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    target_col = "PlacementStatus" if "PlacementStatus" in df.columns else None
    target_dist = df[target_col].value_counts().to_dict() if target_col else {}
    
    summary = {
        "dataset_name": os.path.basename(raw_path),
        "total_rows": rows,
        "total_columns": cols,
        "columns": col_names,
        "data_types": data_types,
        "missing_values": missing_vals,
        "duplicate_rows": dup_rows,
        "unique_counts": unique_counts,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_column": target_col,
        "target_distribution": target_dist
    }
    
    print("=== 10K DATASET INSPECTION SUMMARY ===")
    print(f"Shape: {rows} rows, {cols} columns")
    print(f"Duplicates: {dup_rows}")
    print(f"Missing Values: {missing_vals}")
    print(f"Target Distribution: {target_dist}")
    
    return summary

if __name__ == '__main__':
    audit_and_inspect_dataset("c:/Ascentra/backend/data/raw/placement_10k.csv")
