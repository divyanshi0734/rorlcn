"""One-time migration script to extract historical training telemetry from task-346.log.

Parses the complete 0-51,200 step run log and backfills:
  1. checkpoints/collapse_events.csv
  2. checkpoints/training_telemetry.csv

Going forward, all training resume operations, callbacks, and analyses interact
exclusively with these CSV files.
"""

import os
import re
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = "/Users/divyanshikhare/.gemini/antigravity-ide/brain/94d7175c-e30e-4602-b66b-fadb5d574b36/.system_generated/tasks/task-346.log"
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def migrate_log():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log file not found at: {LOG_PATH}")

    collapse_rows = []
    telemetry_rows = []

    # Regex patterns
    # Pattern for progress lines:
    # Step  10946 | Ep  4500 | Deliv Rate:  32.0% (Peak:  34.0%) | Mean Ret: -11.00 | Mean Hops:  2.4 | Invalid: 0
    prog_re = re.compile(
        r"Step\s+(\d+)\s+\|\s+Ep\s+(\d+)\s+\|\s+Deliv Rate:\s+([\d\.]+)%\s+\(Peak:\s+([\d\.]+)%\)\s+\|\s+Mean Ret:\s+([-\d\.]+)\s+\|\s+Mean Hops:\s+([\d\.]+)\s+\|\s+Invalid:\s+(\d+)"
    )

    # Pattern for collapse warnings:
    # >>> [POLICY REGRESSION WARNING] Step 14045, Episode 5851: Rolling delivery rate (24.0%) dropped 21.0% below running peak (45.0%)! <<<
    collapse_re = re.compile(
        r"Step\s+(\d+),\s+Episode\s+(\d+):\s+Rolling delivery rate\s+\(([\d\.]+)%\)\s+dropped\s+([\d\.]+)%\s+below running peak\s+\(([\d\.]+)%\)"
    )

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            pm = prog_re.search(line)
            if pm:
                telemetry_rows.append({
                    "step": int(pm.group(1)),
                    "episode": int(pm.group(2)),
                    "delivery_rate": float(pm.group(3)) / 100.0,
                    "peak_rate": float(pm.group(4)) / 100.0,
                    "mean_return": float(pm.group(5)),
                    "mean_hops": float(pm.group(6)),
                    "invalid_count": int(pm.group(7)),
                })

            cm = collapse_re.search(line)
            if cm:
                collapse_rows.append({
                    "step": int(cm.group(1)),
                    "episode": int(cm.group(2)),
                    "current_rate": float(cm.group(3)) / 100.0,
                    "drop_pct": float(cm.group(4)),
                    "peak_rate": float(cm.group(5)) / 100.0,
                })

    # Group contiguous collapse events into distinct incidents (gap <= 5 episodes)
    incident_id = 1
    for i, row in enumerate(collapse_rows):
        if i == 0:
            row["incident_id"] = incident_id
        else:
            prev_ep = collapse_rows[i - 1]["episode"]
            if row["episode"] - prev_ep > 5:
                incident_id += 1
            row["incident_id"] = incident_id

    collapse_df = pd.DataFrame(collapse_rows)
    telemetry_df = pd.DataFrame(telemetry_rows)

    collapse_csv = os.path.join(CHECKPOINTS_DIR, "collapse_events.csv")
    telemetry_csv = os.path.join(CHECKPOINTS_DIR, "training_telemetry.csv")

    collapse_df.to_csv(collapse_csv, index=False)
    telemetry_df.to_csv(telemetry_csv, index=False)

    print(f"[*] Migration Complete:")
    print(f"    - Parsed {len(collapse_df)} collapse events into {incident_id} distinct incidents -> {collapse_csv}")
    print(f"    - Parsed {len(telemetry_df)} periodic telemetry checkpoints -> {telemetry_csv}")


if __name__ == "__main__":
    migrate_log()
