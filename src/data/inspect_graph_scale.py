import glob
import os
import pandas as pd

DB = "processed_tables/db"

def inspect():
    files = sorted(glob.glob(f"{DB}/*.parquet"))
    if not files:
        print(f"No parquet files found in {DB}")
        return

    print(f"{'Table':<20} {'Total Rows':<12} {'Unique PKs':<12}")
    print("-" * 46)
    
    total_records = 0
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        total_records += len(df)
        pk_col = df.columns[0]
        unique_pks = df[pk_col].nunique()
        print(f"{name:<20} {len(df):<12,d} {unique_pks:<12,d}")

    print("-" * 46)
    print(f"Total Database Rows: {total_records:,}")

if __name__ == "__main__":
    inspect()
    