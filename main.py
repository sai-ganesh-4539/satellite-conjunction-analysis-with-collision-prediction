import requests

url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
data = requests.get(url).text

with open("active_satellites.tle", "w") as f:
    f.write(data)

lines = data.strip().split('\n')
print(f"Total lines: {len(lines)}")
print(f"Total satellites: {len(lines) // 3}")
print(f"\nFirst 3 satellites:")
for i in range(0, 9, 3):
    print(f"  → {lines[i].strip()}")