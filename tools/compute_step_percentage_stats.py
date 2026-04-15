import json
from pathlib import Path
from collections import defaultdict
import statistics
import csv

BASE = Path(__file__).parent.parent / "data" / "ground_truth_sample"
OUT = Path(__file__).parent.parent / "output" / "step_percentage_stats.csv"

def collect_percentages():
    percentages = defaultdict(list)  # step_id -> list of percent values
    files = sorted(BASE.glob('*.json'))
    for p in files:
        data = json.load(open(p, 'r', encoding='utf-8'))
        total = data.get('total_duration_sec')
        if total is None or total == 0:
            # fallback: compute max end time from procedure_steps
            steps = data.get('procedure_steps', [])
            ends = [s.get('end_sec') for s in steps if s.get('end_sec') is not None]
            total = max(ends) if ends else None
        if total is None or total == 0:
            continue
        for s in data.get('procedure_steps', []):
            sid = s.get('step_id')
            start = s.get('start_sec')
            end = s.get('end_sec')
            if sid is None or start is None or end is None:
                continue
            dur = float(end) - float(start)
            pct = 100.0 * dur / float(total)
            percentages[int(sid)].append(pct)
    return percentages


def summarize(percentages):
    rows = []
    for sid in sorted(percentages.keys()):
        vals = percentages[sid]
        cnt = len(vals)
        mean = statistics.mean(vals) if cnt else 0.0
        mn = min(vals) if cnt else 0.0
        mx = max(vals) if cnt else 0.0
        std = statistics.stdev(vals) if cnt > 1 else 0.0
        rows.append((sid, cnt, mean, mn, mx, std))
    return rows


def write_csv(rows, out_path=OUT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['step_id','count','mean_pct','min_pct','max_pct','std_pct'])
        for r in rows:
            w.writerow(r)
    return out_path


def main():
    percentages = collect_percentages()
    if not percentages:
        print('No data found')
        return
    rows = summarize(percentages)
    out = write_csv(rows)
    print(f'Wrote CSV to: {out}')
    for sid, cnt, mean, mn, mx, std in rows:
        print(f'Step {sid}: count={cnt}, mean={mean:.3f}%, min={mn:.3f}%, max={mx:.3f}%, std={std:.3f}%')

if __name__ == "__main__":
    main()
