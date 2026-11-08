import os
import time
import urllib.request

# Folder where raw .json.gz files will be saved
output_dir = "raw_data"
os.makedirs(output_dir, exist_ok=True)

# Standard browser User-Agent header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Start with a test range (e.g., Oct 01 to Oct 14)
start_day = 1
end_day = 15

for day in range(start_day, end_day + 1):
    for hour in range(24):
        filename = f"2026-10-{day:02d}-{hour}.json.gz"
        url = f"https://data.gharchive.org/{filename}"
        dest_path = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[SKIP] {filename} exists.")
            continue

        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"[OK] Saved {filename}")
        except urllib.error.HTTPError as e:
            print(f"[ERROR {e.code}] Could not fetch {filename}")
        except Exception as e:
            print(f"[FAILED] {filename}: {e}")

        # Brief pause between requests to prevent rate limiting
        time.sleep(0.5)