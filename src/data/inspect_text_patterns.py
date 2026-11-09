import os
import re
import pandas as pd

DB = "processed_tables/db"

def inspect_field(df: pd.DataFrame, table_name: str, col_name: str, n_samples: int = 5):
    if col_name not in df.columns:
        return
    
    series = df[col_name].dropna().astype(str)
    if len(series) == 0:
        print(f"[{table_name}.{col_name}] All values are null.\n")
        return

    print("=" * 80)
    print(f"TABLE: {table_name} | COLUMN: {col_name} (Non-null: {len(series):,})")
    print("=" * 80)
    
    # Character length statistics
    lengths = series.str.len()
    print(f"Length Stats -> Min: {lengths.min()} | Median: {lengths.median():.0f} | Max: {lengths.max()}")
    
    # Common preserved punctuation/structural symbols
    symbols = ['#', ':', '(', ')', '/', '|', '@', '`', '\n', '-', '+']
    symbol_counts = {s: int(series.str.contains(re.escape(s)).sum()) for s in symbols}
    print(f"Symbol Presence (% of rows):")
    for s, cnt in symbol_counts.items():
        pct = (cnt / len(series)) * 100
        disp_s = "\\n" if s == '\n' else s
        print(f"  '{disp_s}': {pct:5.1f}%", end=" | ")
    print("\n")

    # Print 5 diverse full samples (short, median, long)
    sorted_series = series.iloc[lengths.argsort()]
    sample_indices = [
        0,                                   # shortest
        len(sorted_series) // 4,            # 25th percentile
        len(sorted_series) // 2,            # median
        (3 * len(sorted_series)) // 4,      # 75th percentile
        len(sorted_series) - 1              # longest
    ]
    
    print(f"--- Full Uncut Samples (from shortest to longest) ---")
    for rank, idx in zip(["Min", "Q1", "Median", "Q3", "Max"], sample_indices):
        text = sorted_series.iloc[idx]
        # Show first 300 characters if text is extremely long
        preview = text if len(text) <= 300 else text[:300] + f" ... [truncated total {len(text)} chars]"
        print(f"\n[{rank} length: {len(text)} chars]:\n{preview}")
    print("\n")


def main():
    targets = [
        ("PullRequests", ["title", "head_ref", "base_ref"]),
        ("Issues", ["title", "body", "labels"]),
        ("IssueComments", ["body"]),
        ("PRReviews", ["body"]),
        ("PRReviewComments", ["body", "diff_hunk", "path"]),
    ]
    
    for table_name, cols in targets:
        path = os.path.join(DB, f"{table_name}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for c in cols:
            inspect_field(df, table_name, c)

if __name__ == "__main__":
    main()
    