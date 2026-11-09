import glob
import os
import pandas as pd

DB = "processed_tables/db"

def inspect_table_columns():
    files = sorted(glob.glob(f"{DB}/*.parquet"))
    if not files:
        print(f"No parquet files found in {DB}")
        return

    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        print("=" * 80)
        print(f"TABLE: {name} (Total Rows: {len(df):,})")
        print("=" * 80)
        
        # Summary table of each column
        summary = []
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)
            
            # Grab 2 non-null sample values to see what they look like
            samples = df[col].dropna().head(2).tolist()
            sample_str = str(samples) if samples else "All Null"
            if len(sample_str) > 35:
                sample_str = sample_str[:32] + "..."

            summary.append({
                "Column": col,
                "Dtype": dtype,
                "Null %": f"{null_pct:.1f}%",
                "Unique": n_unique,
                "Sample": sample_str
            })
            
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        print("\n")

if __name__ == "__main__":
    inspect_table_columns()
    