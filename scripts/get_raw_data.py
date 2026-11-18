import os
import time
import urllib.error
from datetime import datetime, timedelta
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Folder where raw .json.gz files will be saved
OUTPUT_DIR = "raw_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define your exact start and end dates
# Dec 1, 2025 to March 21, 2026.
# Truncated from the documented full corpus (Oct 31, 2026) because this box only
# has ~150 GB free on /train-data-5-hdd: 2025-12-01 .. 2026-10-31 is 8,042 hourly
# files at a measured ~35.7 MB each, i.e. ~221 GB beyond what is already here.
# 2026-03-21 is the last date that buys anything: docs/VALIDATION_DATA.md §1 makes
# 2026-02-15 the approved validation cutoff (second choice 2026-03-08) and 2026-03-14
# the latest cutoff with a fully usable 7-day window for all six activities, so its
# label window ends 2026-03-21. Past that no 7-day validation window exists, so the
# remaining months would only add training anchors.
# activity types appear at plausible density (~18k PullRequestEvents/hour, vs
# ~1.3k in Sep 2026 and ~110 in Aug 2026), so Dec 2025 doubles as the reference month
# for scripts/verify_raw_data.py.
# November 1 until now is excluded because it seems there was data corruption during that time
START_DATE = datetime(2025, 12, 1).date()
END_DATE   = datetime(2026, 3, 21).date()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def generate_file_list(start_date, end_date):
    """Generates all hourly filenames correctly handling calendar months and days."""
    files = []
    curr = start_date
    while curr <= end_date:
        for hour in range(24):
            filename = f"{curr.strftime('%Y-%m-%d')}-{hour}.json.gz"
            files.append(filename)
        curr += timedelta(days=1)
    return files

def download_file(filename):
    dest_path = os.path.join(OUTPUT_DIR, filename)
    url = f"https://data.gharchive.org/{filename}"

    # Skip if valid file already exists (> 1KB to avoid corrupted partial files)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        return f"[SKIP] {filename}"

    # Download to a temporary file first, then rename (atomic save)
    temp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
            f.write(resp.read())
        
        # Verify downloaded size
        if os.path.getsize(temp_path) > 0:
            os.replace(temp_path, dest_path)
            return f"[OK] {filename}"
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return f"[EMPTY] {filename}"
            
    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # 404 indicates missing data from GH Archive
        return f"[HTTP {e.code}] {filename}"
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"[ERROR] {filename}: {e}"

def main():
    target_files = generate_file_list(START_DATE, END_DATE)
    print(f"Total files to verify/download: {len(target_files)}")
    
    # Download with 4 parallel threads to significantly speed up network transfer
    # (Do not use too many workers to avoid being rate-limited by GH Archive)
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(download_file, target_files):
            if not res.startswith("[SKIP]"):
                print(res)

if __name__ == "__main__":
    main()