import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

BASE = Path(__file__).parent.parent / "data" / "ground_truth_sample"
OUT_DIR = Path(__file__).parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# parameters
WINDOW = 5.0  # seconds

# collect error timestamps
timestamps = []
max_t = 0.0
for p in sorted(BASE.glob('*.json')):
    try:
        data = json.load(open(p, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"Skipping {p.name}: {e}")
        continue
    events = data.get('events', [])
    for ev in events:
        if ev.get('type') == 'error_detected':
            t = ev.get('timestamp_sec')
            if t is None:
                continue
            try:
                t = float(t)
            except Exception:
                continue
            timestamps.append(t)
            if t > max_t:
                max_t = t

if not timestamps:
    print('No error events found.')
    raise SystemExit(0)

# create bins
num_bins = int(np.ceil((max_t + 1e-6) / WINDOW))
bins = np.arange(0, (num_bins + 1) * WINDOW, WINDOW)
counts, edges = np.histogram(timestamps, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2.0

# save CSV
csv_path = OUT_DIR / 'error_timeline_counts.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['window_start_sec','window_end_sec','count'])
    for i in range(len(counts)):
        w.writerow([edges[i], edges[i+1], int(counts[i])])

# plot
plt.figure(figsize=(12,4))
plt.bar(centers, counts, width=WINDOW*0.9, align='center')
plt.xlabel('Time (s)')
plt.ylabel('Error count (all files)')
plt.title(f'Error events per {int(WINDOW)}s window (aggregated)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
img_path = OUT_DIR / 'error_timeline.png'
plt.savefig(img_path)
plt.close()

print(f'Wrote CSV: {csv_path}')
print(f'Wrote PNG: {img_path}')
