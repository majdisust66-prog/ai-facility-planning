"""
make_sample_history.py - a second SYNTHETIC history (different seed and names) for trying the upload.
Run:  python make_sample_history.py
"""
import os

import pandas as pd

from generate_synthetic_data import generate

os.chdir(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = ["wo_id", "created_at", "assigned_at", "completed_at", "status", "priority", "scope",
           "facility_id", "region", "contract_id", "contractor_id", "asset_id"]
REGIONS = {"Region-1": "North", "Region-2": "South", "Region-3": "East",
           "Region-4": "West", "Region-5": "Central", "Region-6": "Coastal"}


def main():
    df, _ = generate(9000, seed=7)
    df["wo_id"] = df.wo_id.str.replace("WO-", "DEMO-")
    for col, old, new in [("facility_id", "FAC-", "SITE-"), ("asset_id", "FAC-", "SITE-"),
                          ("contractor_id", "CTR-", "VENDOR-"), ("contract_id", "CON-", "AGR-")]:
        df[col] = df[col].str.replace(old, new)
    df["region"] = df.region.map(REGIONS)
    for col in ["created_at", "assigned_at", "completed_at"]:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
    df = df[COLUMNS]
    # three deliberately wrong rows, so rejection with a reason can be shown
    bad = df.head(3).copy()
    bad["wo_id"] = ["BAD-001", "BAD-002", "BAD-003"]
    bad.iloc[0, bad.columns.get_loc("created_at")] = "05/09/2025"
    bad.iloc[1, bad.columns.get_loc("priority")] = "Urgent"
    bad.iloc[2, bad.columns.get_loc("facility_id")] = None
    os.makedirs("samples", exist_ok=True)
    pd.concat([df, bad]).to_csv("samples/history_sample.csv", index=False)
    df.head(3).to_csv("samples/history_template.csv", index=False)
    print("rows:", len(df) + len(bad), "| size MB:",
          round(os.path.getsize("samples/history_sample.csv") / 1e6, 2))


if __name__ == "__main__":
    main()
