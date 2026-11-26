"""
generate_synthetic_data.py - SYNTHETIC Facility Management work-order generator.

ALL OUTPUT IS ARTIFICIAL. Relationships (contractor skill, backlog effect, etc.)
are planted by this script and listed in generation_log.json. A model that
"finds" them proves the METHOD works; it says nothing about any real contract.

Usage:
    python generate_synthetic_data.py --out synthetic_data --n 24000 --seed 42
Requirements: pip install pandas numpy
"""
import argparse
import heapq
import json
import os

import numpy as np
import pandas as pd

# ----------------------------- configuration ------------------------------
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2026-06-30 23:59")
EXTRACT_DATE = pd.Timestamp("2026-06-30 12:00")  # work orders unfinished at this date are "open"

REGIONS = ["Region-1", "Region-2", "Region-3", "Region-4", "Region-5", "Region-6"]
CONTRACT_REGIONS = {"CON-01": ["Region-1", "Region-2"],
                    "CON-02": ["Region-3", "Region-4"],
                    "CON-03": ["Region-5", "Region-6"]}
SCOPES = ["HVAC", "Electrical", "Plumbing", "Civil", "Fire Safety", "Elevators"]
SCOPE_GROUP = {"HVAC": "MEP", "Electrical": "MEP", "Plumbing": "MEP",
               "Civil": "Civil", "Fire Safety": "Safety", "Elevators": "Safety"}
SCOPE_WEIGHT = [0.30, 0.22, 0.18, 0.12, 0.10, 0.08]
FACILITY_TYPES = ["Office", "Hospital", "School", "Mall", "Warehouse"]

# priority: (label, response SLA h, completion SLA h, median completion h, share)
PRIORITY = {"P1-Emergency": (1, 4, 2.0, 0.07),
            "P2-High": (4, 24, 12.0, 0.23),
            "P3-Medium": (8, 72, 40.0, 0.45),
            "P4-Low": (24, 168, 100.0, 0.25)}

# planted effects (multipliers on completion duration)
CONTRACTOR_SKILL = {"CTR-01": 0.85, "CTR-02": 1.00, "CTR-03": 1.35, "CTR-04": 0.95, "CTR-05": 1.15}
REGION_EFFECT = {"Region-1": 1.00, "Region-2": 1.05, "Region-3": 0.95,
                 "Region-4": 1.25, "Region-5": 1.00, "Region-6": 1.10}
BACKLOG_PER_OPEN_WO = 0.02    # +2% duration per open WO of the same contractor (capped at 25 open WOs)
AFTER_HOURS_EFFECT = 1.30     # created 18:00-06:00 or Friday/Saturday
HVAC_SUMMER_EFFECT = 1.20     # HVAC in Jun-Aug
REPEAT_EFFECT = 1.15          # repeat fault on same asset within 30 days
NOISE_SIGMA = 0.60
DURATION_SCALE = 0.50         # calibration so overall breach rate is plausible (~20-30%)


def build_facilities(rng):
    rows = []
    for i in range(1, 41):
        region = REGIONS[(i - 1) % 6]
        contract = next(c for c, r in CONTRACT_REGIONS.items() if region in r)
        rows.append({"facility_id": f"FAC-{i:03d}", "region": region, "contract_id": contract,
                     "facility_type": rng.choice(FACILITY_TYPES, p=[.3, .1, .2, .2, .2]),
                     "area_m2": int(rng.integers(800, 25000)),
                     "building_age_years": int(rng.integers(2, 40)),
                     "facility_factor": float(np.round(rng.lognormal(0, 0.15), 3))})
    return pd.DataFrame(rows)


def contractor_map(rng):
    """Each (contract, scope group) is served by one contractor."""
    ctrs = list(CONTRACTOR_SKILL)
    slots = [(c, g) for c in CONTRACT_REGIONS for g in sorted(set(SCOPE_GROUP.values()))]
    order = rng.permutation(len(slots))  # cyclic assignment so every contractor is used
    return {slots[j]: ctrs[k % len(ctrs)] for k, j in enumerate(order)}


def generate(n, seed):
    rng = np.random.default_rng(seed)
    fac = build_facilities(rng)
    cmap = contractor_map(rng)

    # creation times: weekday-heavy, business hours-heavy
    span = (END - START).total_seconds()
    t = START + pd.to_timedelta(rng.uniform(0, span, n), unit="s")
    t = pd.Series(t).sort_values().reset_index(drop=True)

    pr_labels = list(PRIORITY)
    pr = rng.choice(pr_labels, size=n, p=[PRIORITY[p][3] for p in pr_labels])
    scope = rng.choice(SCOPES, size=n, p=SCOPE_WEIGHT)
    fac_idx = rng.choice(len(fac), size=n, p=_zipf(len(fac), 0.6, rng))
    asset_no = rng.choice(15, size=n, p=_zipf(15, 1.0, rng))

    rows, last_seen, heaps = [], {}, {c: [] for c in CONTRACTOR_SKILL}
    for i in range(n):
        f = fac.iloc[fac_idx[i]]
        ctr = cmap[(f.contract_id, SCOPE_GROUP[scope[i]])]
        ts = t[i]
        # backlog = this contractor's WOs still open at creation time (known at prediction point)
        h = heaps[ctr]
        while h and h[0] <= ts:
            heapq.heappop(h)
        backlog = len(h)

        asset = f"{f.facility_id}-A{asset_no[i]:02d}"
        key = (asset, scope[i])
        repeat = key in last_seen and (ts - last_seen[key]).days <= 30
        last_seen[key] = ts

        after_hours = ts.hour >= 18 or ts.hour < 6 or ts.day_of_week in (4, 5)
        mult = (CONTRACTOR_SKILL[ctr] * REGION_EFFECT[f.region] * f.facility_factor
                * (1 + BACKLOG_PER_OPEN_WO * min(backlog, 25))
                * (AFTER_HOURS_EFFECT if after_hours else 1.0)
                * (HVAC_SUMMER_EFFECT if scope[i] == "HVAC" and ts.month in (6, 7, 8) else 1.0)
                * (REPEAT_EFFECT if repeat else 1.0))
        resp_sla, comp_sla, median, _ = PRIORITY[pr[i]]
        dur = median * DURATION_SCALE * mult * rng.lognormal(0, NOISE_SIGMA)
        resp = min(resp_sla * rng.lognormal(-0.3, 0.7), dur * 0.5)
        start_work = resp + rng.uniform(0.2, 2.0)
        start_work = min(start_work, dur * 0.9)

        finish = ts + pd.Timedelta(hours=dur)
        heapq.heappush(h, finish)
        rows.append({
            "wo_id": f"WO-{i + 1:06d}", "wo_type": "Corrective",
            "created_at": ts, "assigned_at": ts + pd.Timedelta(hours=resp),
            "actual_start": ts + pd.Timedelta(hours=start_work), "completed_at": finish,
            "priority": pr[i], "scope": scope[i],
            "fault_code": f"{SCOPE_GROUP[scope[i]][:3].upper()}-{int(rng.integers(1, 9)):02d}",
            "facility_id": f.facility_id, "region": f.region, "contract_id": f.contract_id,
            "contractor_id": ctr, "asset_id": asset,
            "reopened": bool(rng.random() < 0.04),
            "status": "Completed",
            # --- hidden analytical columns, dropped from the raw export ---
            "_comp_sla_h": comp_sla, "_resp_sla_h": resp_sla,
        })
    df = pd.DataFrame(rows)

    # censoring: unfinished at extract date -> open; 2% cancelled
    open_mask = df.completed_at > EXTRACT_DATE
    df.loc[open_mask, ["completed_at", "actual_start"]] = pd.NaT
    df.loc[open_mask & (df.assigned_at > EXTRACT_DATE), "assigned_at"] = pd.NaT
    df.loc[open_mask, "status"] = np.where(df.loc[open_mask, "assigned_at"].isna(), "Waiting", "In Progress")
    canc = (~open_mask) & (rng.random(n) < 0.02)
    df.loc[canc, ["assigned_at", "actual_start", "completed_at"]] = pd.NaT
    df.loc[canc, "status"] = "Cancelled"
    return df, fac


def _zipf(k, a, rng):
    w = 1.0 / np.arange(1, k + 1) ** a
    w = rng.permutation(w)
    return w / w.sum()


def inject_defects(df, rng):
    """Make the raw export realistically dirty. Returns dirty df and a defect log."""
    d = df.drop(columns=["_comp_sla_h", "_resp_sla_h"]).copy()
    n, log = len(d), {}

    def pick(frac):
        return rng.random(n) < frac

    for col in ["created_at", "assigned_at", "actual_start", "completed_at"]:
        d[col] = d[col].dt.strftime("%Y-%m-%d %H:%M")
    # mixed date format (dd/mm/yyyy) in 2% of created_at
    m = pick(0.02)
    d.loc[m, "created_at"] = pd.to_datetime(d.loc[m, "created_at"]).dt.strftime("%d/%m/%Y %H:%M")
    log["created_at_alt_format"] = int(m.sum())
    # inconsistent priority labels
    m = pick(0.06)
    d.loc[m, "priority"] = d.loc[m, "priority"].str.lower()
    log["priority_lowercase"] = int(m.sum())
    m = pick(0.03)
    d.loc[m, "priority"] = d.loc[m, "priority"].str.split("-").str[0]
    log["priority_short_code"] = int(m.sum())
    # region/scope whitespace and case variants
    m = pick(0.04)
    d.loc[m, "region"] = " " + d.loc[m, "region"].str.upper() + " "
    log["region_case_whitespace"] = int(m.sum())
    m = pick(0.03)
    d.loc[m, "scope"] = d.loc[m, "scope"].str.replace("Fire Safety", "Fire safety").str.replace("HVAC", "hvac")
    log["scope_variants"] = int(m.sum())
    # missing values
    for col, frac in [("fault_code", 0.05), ("asset_id", 0.03), ("actual_start", 0.04)]:
        m = pick(frac)
        d.loc[m, col] = np.nan
        log[f"missing_{col}_extra"] = int(m.sum())
    # completed before created (swap)
    idx = d.index[(d.status == "Completed") & pick(0.003)]
    d.loc[idx, ["created_at", "completed_at"]] = d.loc[idx, ["completed_at", "created_at"]].values
    log["completed_before_created"] = int(len(idx))
    # future date typo (year +1)
    idx = d.index[pick(0.002)]
    d.loc[idx, "created_at"] = d.loc[idx, "created_at"].astype(str).str.replace("2025", "2027").str.replace("2026", "2027")
    log["future_created_date_typo"] = int(len(idx))
    # completed work orders with missing completed_at
    idx = d.index[(d.status == "Completed") & pick(0.01)]
    d.loc[idx, "completed_at"] = np.nan
    log["completed_status_without_date"] = int(len(idx))
    # duplicated rows
    dup = d.sample(frac=0.005, random_state=int(rng.integers(0, 1e6)))
    d = pd.concat([d, dup], ignore_index=True)
    log["duplicate_rows_appended"] = int(len(dup))
    d = d.sample(frac=1, random_state=1).reset_index(drop=True)
    d["data_origin"] = "SYNTHETIC"
    return d, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed + 1)

    df, fac = generate(a.n, a.seed)
    raw, defects = inject_defects(df, rng)

    # ground-truth breach rate on the CLEAN data (closed WOs only) for later sanity checks
    closed = df[df.status == "Completed"]
    hrs = (closed.completed_at - closed.created_at).dt.total_seconds() / 3600
    breach = hrs > closed._comp_sla_h
    truth = {"closed_work_orders": int(len(closed)), "breach_rate_clean": round(float(breach.mean()), 4),
             "breach_rate_by_priority": breach.groupby(closed.priority).mean().round(3).to_dict(),
             "breach_rate_by_contractor": breach.groupby(closed.contractor_id).mean().round(3).to_dict(),
             "open_or_waiting": int(df.status.isin(["In Progress", "Waiting"]).sum())}

    raw.to_csv(os.path.join(a.out, "work_orders_raw.csv"), index=False, encoding="utf-8-sig")
    fac.drop(columns=["facility_factor"]).to_csv(os.path.join(a.out, "facilities.csv"), index=False)
    sla = pd.DataFrame([{"priority": p, "response_sla_h": v[0], "completion_sla_h": v[1]}
                        for p, v in PRIORITY.items()])
    sla.to_csv(os.path.join(a.out, "sla_matrix.csv"), index=False)

    log = {"disclosure": "SYNTHETIC DATA - no real organisation, contractor or facility",
           "seed": a.seed, "n_requested": a.n, "extract_date": str(EXTRACT_DATE),
           "planted_effects": {"contractor_skill": CONTRACTOR_SKILL, "region_effect": REGION_EFFECT,
                               "backlog_per_open_wo": BACKLOG_PER_OPEN_WO,
                               "after_hours": AFTER_HOURS_EFFECT, "hvac_summer": HVAC_SUMMER_EFFECT,
                               "repeat_within_30d": REPEAT_EFFECT, "noise_sigma": NOISE_SIGMA, "duration_scale": DURATION_SCALE,
                               "facility_factor": "hidden per-facility lognormal(0, 0.15)"},
           "injected_defects": defects, "clean_truth": truth}
    with open(os.path.join(a.out, "generation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(a.out, "DISCLOSURE.txt"), "w", encoding="utf-8") as f:
        f.write("SYNTHETIC DATA. Generated by generate_synthetic_data.py. Not real. "
                "Any findings validate the methodology only, not real contract performance.\n")
    print(f"rows written: {len(raw)} | clean breach rate: {truth['breach_rate_clean']}")
    print(json.dumps(truth["breach_rate_by_contractor"]))


if __name__ == "__main__":
    main()
