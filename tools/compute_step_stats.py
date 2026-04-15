import json
from pathlib import Path
from collections import defaultdict
import statistics
import csv

BASE = Path(__file__).parent.parent / "data" / "ground_truth_sample"
OUT = Path(__file__).parent.parent / "output" / "step_stats.csv"


def collect_durations():
    durations = defaultdict(list)  # step_id -> list of durations
    files = sorted(BASE.glob("*.json"))
    for p in files:
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            continue
        steps = data.get("procedure_steps", [])
        for s in steps:
            sid = s.get("step_id")
            start = s.get("start_sec")
            end = s.get("end_sec")
            if sid is None or start is None or end is None:
                continue
            try:
                dur = float(end) - float(start)
            except Exception:
                continue
            durations[int(sid)].append(dur)
    return durations


def summarize(durations):
    rows = []
    for sid in sorted(durations.keys()):
        vals = durations[sid]
        cnt = len(vals)
        mean = statistics.mean(vals) if cnt else 0.0
        mn = min(vals) if cnt else 0.0
        mx = max(vals) if cnt else 0.0
        std = statistics.stdev(vals) if cnt > 1 else 0.0
        rows.append((sid, cnt, mean, mn, mx, std))
    return rows


def write_csv(rows, out_path=OUT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_id", "count", "mean_sec", "min_sec", "max_sec", "std_sec"])
        for r in rows:
            w.writerow(r)
    return out_path


def main():
    durations = collect_durations()
    if not durations:
        print("No procedure steps found in ground truth samples.")
        return
    rows = summarize(durations)
    out = write_csv(rows)
    print(f"Wrote CSV to: {out}")
    print("Summary:")
    for sid, cnt, mean, mn, mx, std in rows:
        print(f"Step {sid}: count={cnt}, mean={mean:.3f}s, min={mn:.3f}s, max={mx:.3f}s, std={std:.3f}s")


if __name__ == '__main__':
    main()
