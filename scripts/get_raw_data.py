import os
import time
from datetime import datetime, timedelta
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Folder where raw .json.gz files will be saved
OUTPUT_DIR = "raw_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define your exact start and end dates
# Sept 1, 2026 to Nov 8, 2026
START_DATE = datetime(2026, 9, 1).date()
END_DATE   = datetime(2026, 11, 8).date()

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