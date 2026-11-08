import os
import glob
import pandas as pd  # or pyarrow.parquet as pq

# Folder where your preprocess.py saved the tables
TABLES_DIR = "processed_tables\db"

# The schema definition we used
EXPECTED_SCHEMA = {
    "Users": {"pk": "user_id", "fks": []},
    "Repositories": {"pk": "repo_id", "fks": []},
    "PullRequests": {"pk": "pr_id", "fks": ["author_id", "repo_id", "merged_by_id"]},
    "Issues": {"pk": "issue_id", "fks": ["author_id", "repo_id", "assignee_id", "pull_request_id"]},
    "IssueComments": {"pk": "comment_id", "fks": ["author_id", "repo_id", "issue_id"]},
    "PRReviews": {"pk": "review_id", "fks": ["reviewer_id", "repo_id", "pr_id"]},
    "PRReviewComments": {"pk": "comment_id", "fks": ["author_id", "repo_id", "pr_id", "pull_request_review_id", "in_reply_to_id"]},
    "Pushes": {"pk": "push_id", "fks": ["user_id", "repo_id"]}
}

print(f"Scanning tables in: {TABLES_DIR} ...\n")

for table_name, keys in EXPECTED_SCHEMA.items():
    # Find matching parquet or csv file
    matches = glob.glob(os.path.join(TABLES_DIR, f"{table_name}.*"))
    if not matches:
        # Check lowercase just in case
        matches = glob.glob(os.path.join(TABLES_DIR, f"{table_name.lower()}.*"))
        
    if not matches:
        print(f"❌ Table file for '{table_name}' NOT found in {TABLES_DIR}")
        continue
        
    file_path = matches[0]
    
    # Read just the header
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, nrows=1)
        
    cols = set(df.columns)
    
    # Verify Primary Key
    pk_ok = keys["pk"] in cols
    # Verify Foreign Keys
    missing_fks = [fk for fk in keys["fks"] if fk not in cols]
    
    status = "✅ PASS" if pk_ok and not missing_fks else "❌ MISMATCH"
    print(f"[{status}] Table: {table_name}")
    if not pk_ok:
        print(f"   Missing PK: {keys['pk']}")
    if missing_fks:
        print(f"   Missing FKs: {missing_fks}")

print("\nVerification complete!")