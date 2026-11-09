import os
import requests
from requests.auth import HTTPBasicAuth
import csv
import time
from dotenv import load_dotenv
load_dotenv()

USERNAME = os.environ.get("SPACE_TRACK_USER", "")
PASSWORD = os.environ.get("SPACE_TRACK_PASS", "")

if not USERNAME or not PASSWORD:
    raise SystemExit(
        "Missing Space-Track credentials. Create a .env file next to this script:\n"
        "  SPACE_TRACK_USER=your_email\n"
        "  SPACE_TRACK_PASS=your_password\n"
        "NEVER commit .env to git (it is in .gitignore)."
    )

BASE_URL = "https://www.space-track.org"

session = requests.Session()

# Step 1: Login
print("Logging in to Space-Track...")
login = session.post(f"{BASE_URL}/ajaxauth/login", data={
    "identity": USERNAME,
    "password": PASSWORD
})

if login.status_code != 200:
    print("Login failed! Check your email/password.")
    print(f"Status: {login.status_code}")
    print(login.text)
else:
    print("Login successful!")

# Step 2: Download CDM data (public conjunctions)
print("\nDownloading CDMs...")

# CDM public data
cdm_url = f"{BASE_URL}/basicspacedata/query/class/cdm_public/orderby/TCA%20desc/limit/3000/format/csv"

response = session.get(cdm_url)

if response.status_code == 200:
    with open(os.path.join("data", "cdms.csv"), "w", newline='') as f:
        f.write(response.text)
    
    # Count rows
    rows = response.text.strip().split('\n')
    print(f"Saved {len(rows)-1} CDM records to cdms.csv")
    print(f"First line (headers): {rows[0][:200]}")
else:
    print(f"Failed: {response.status_code}")
    print(response.text[:500])

# Step 3: Also download TLE data from Space-Track (no VPN needed!)
print("\nDownloading TLEs from Space-Track...")

tle_url = f"{BASE_URL}/basicspacedata/query/class/tle_latest/orderby/NORAD_CAT_ID/format/tle"

response2 = session.get(tle_url)

if response2.status_code == 200:
    with open(os.path.join("data", "spacetrack_tle.tle"), "w") as f:
        f.write(response2.text)
    
    lines = response2.text.strip().split('\n')
    print(f"Saved {len(lines)//3} TLE records")
else:
    print(f"TLE download failed: {response2.status_code}")

# Step 4: Download SATCAT (satellite catalog)
print("\nDownloading SATCAT...")
satcat_url = f"{BASE_URL}/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/format/csv"

response3 = session.get(satcat_url)

if response3.status_code == 200:
    with open(os.path.join("data", "satcat.csv"), "w", newline='') as f:
        f.write(response3.text)
    rows = response3.text.strip().split('\n')
    print(f"Saved {len(rows)-1} SATCAT records")
else:
    print(f"SATCAT failed: {response3.status_code}")

print("\nDone! Check data/")