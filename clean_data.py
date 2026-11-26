"""
clean_data.py - Module 1: cleaning of the raw work-order export.

Principles
  1. Never silently delete: every rule writes a row to cleaning_log.csv (rows affected, action, rationale).
  2. Standardise only when the mapping is unambiguous. Do not "repair" values we cannot verify
     (e.g. swapped dates) - flag them and exclude them from model labels.
  3. Keep one row per work order; add flags + a `label_status` column that says whether
     (and how) the row may be used as a training label.

Usage:
    python clean_data.py --raw synthetic_data/work_orders_raw.csv --sla synthetic_data/sla_matrix.csv \
        --facilities synthetic_data/facilities.csv --out synthetic_data --extract-date "2026-06-30 12:00"
Requirements: pip install pandas numpy
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

DATE_COLS = ["created_at", "assigned_at", "actual_start", "completed_at"]
PRIORITY_MAP = {"P1": "P1-Emergency", "P2": "P2-High", "P3": "P3-Medium", "P4": "P4-Low"}
SCOPES = ["HVAC", "Electrical", "Plumbing", "Civil", "Fire Safety", "Elevators"]

LOG = []


def log(step, issue, n, action, rationale, implication=""):
    LOG.append({"step": step, "issue": issue, "rows_affected": int(n), "action": action,
                "rationale": rationale, "implication": implication})


def parse_dates(s):
    """Two known formats; ambiguous text is NOT guessed. Returns (parsed, n_alt_format)."""
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")
    alt = pd.to_datetime(s[iso.isna() & s.notna()], format="%d/%m/%Y %H:%M", errors="coerce")
    out = iso.copy()
    out.loc[alt.index] = alt
    return out, int(alt.notna().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sla", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--extract-date", required=True, help="date the data was exported")
    a = ap.parse_args()
    extract = pd.Timestamp(a.extract_date)

    df = pd.read_csv(a.raw, dtype=str, encoding="utf-8-sig")
    n0 = len(df)
    log("0 load", "rows in raw file", n0, "none", "baseline")

    # 1. duplicates ---------------------------------------------------------
    d = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log("1 duplicates", "fully identical rows", d, "dropped (keep first)",
        "exact copies carry no extra information; would double-count KPIs")
    dup_id = int(df["wo_id"].duplicated().sum())
    log("1 duplicates", "same wo_id with different content", dup_id,
        "none needed" if dup_id == 0 else "REVIEW MANUALLY", "wo_id must be unique")
    assert dup_id == 0, "duplicate wo_id with conflicting content - resolve before continuing"

    # 2. categorical standardisation ---------------------------------------
    def clean_priority(x):
        if pd.isna(x):
            return np.nan
        m = re.match(r"\s*[pP]\s*([1-4])", x)
        return PRIORITY_MAP.get(f"P{m.group(1)}") if m else np.nan

    before = df["priority"].copy()
    df["priority"] = before.map(clean_priority)
    log("2 standardise", "priority label variants (case / short code)", (before != df["priority"]).sum(),
        "mapped to P1..P4 canonical labels", "prefix P1-P4 is unambiguous")
    log("2 standardise", "priority unmappable", df["priority"].isna().sum(), "left null", "no guessing")

    before = df["region"].copy()
    df["region"] = before.str.strip().str.title()
    log("2 standardise", "region case / whitespace", (before != df["region"]).sum(), "strip + title-case", "trivial")

    canon = {s.lower(): s for s in SCOPES}
    before = df["scope"].copy()
    df["scope"] = before.str.strip().str.lower().map(canon)
    log("2 standardise", "scope case variants", (before != df["scope"]).sum(), "mapped to canonical list",
        "case-insensitive match against the scope master list")
    log("2 standardise", "scope not in master list", df["scope"].isna().sum(), "left null", "no guessing")

    # 3. dates --------------------------------------------------------------
    for c in DATE_COLS:
        df[c], n_alt = parse_dates(df[c])
        if n_alt:
            log("3 dates", f"{c}: dd/mm/yyyy text instead of ISO", n_alt, "parsed with explicit format",
                "format is known from the source; in a real export confirm day/month order with the data owner",
                "dd/mm vs mm/dd ambiguity is a real risk")
    nat_in_text = int(df["created_at"].isna().sum())
    log("3 dates", "created_at unparseable", nat_in_text, "kept null, excluded", "creation time is mandatory")

    # 4. date validity flags (flag, do not repair) --------------------------
    df["dq_future_created"] = df["created_at"] > extract
    log("4 validity", "created_at after extract date", df["dq_future_created"].sum(),
        "flagged (dq_future_created), excluded from labels",
        "cannot verify the true date; typo repair would be a guess")
    order_cols = [c for c in DATE_COLS if c in df]
    bad = pd.Series(False, index=df.index)
    for x, y in zip(order_cols, order_cols[1:]):
        bad |= (df[y] < df[x])
    # created vs completed directly (skips missing intermediate dates)
    bad |= df["completed_at"] < df["created_at"]
    df["dq_date_order"] = bad
    log("4 validity", "illogical date order (a later stage dated before an earlier one)", bad.sum(),
        "flagged (dq_date_order), excluded from labels",
        "could be swapped fields or entry errors; we cannot tell which")

    # 5. status consistency -------------------------------------------------
    df["dq_completed_no_date"] = (df["status"] == "Completed") & df["completed_at"].isna()
    log("5 consistency", "status=Completed without completed_at", df["dq_completed_no_date"].sum(),
        "flagged, excluded from labels", "resolution time unknown; cannot decide breach")
    df["dq_open_with_end"] = df["status"].isin(["In Progress", "Waiting"]) & df["completed_at"].notna()
    log("5 consistency", "open status but has completed_at", df["dq_open_with_end"].sum(), "flagged", "contradiction")

    # 6. missing values -----------------------------------------------------
    m = df["fault_code"].isna()
    df["fault_code"] = df["fault_code"].fillna("UNKNOWN")
    log("6 missing", "fault_code missing", m.sum(), "filled with 'UNKNOWN' category",
        "missingness itself may be informative; do not invent a code")
    df["dq_asset_missing"] = df["asset_id"].isna()
    log("6 missing", "asset_id missing", df["dq_asset_missing"].sum(), "kept null + flag",
        "recurring-work-order features cannot be built for these rows",
        "recurrence will be under-counted for ~3% of rows")
    log("6 missing", "actual_start missing (non-open, non-cancelled)",
        (df["actual_start"].isna() & df["status"].eq("Completed")).sum(), "kept null",
        "not used as a predictor (unknown at prediction time)")

    # 7. referential checks ------------------------------------------------
    fac = pd.read_csv(a.facilities)
    df = df.merge(fac[["facility_id", "region"]].rename(columns={"region": "_fac_region"}),
                  on="facility_id", how="left")
    miss_fac = df["_fac_region"].isna().sum()
    mismatch = (df["_fac_region"].notna() & (df["_fac_region"] != df["region"])).sum()
    log("7 referential", "facility_id not in facility master", miss_fac, "none" if miss_fac == 0 else "flag", "")
    log("7 referential", "region differs from facility master", mismatch,
        "none" if mismatch == 0 else "use master region", "master data is the source of truth")
    df = df.drop(columns="_fac_region")

    # 8. SLA join + label ---------------------------------------------------
    sla = pd.read_csv(a.sla)
    df = df.merge(sla, on="priority", how="left")
    df["reopened"] = df["reopened"].map({"True": True, "False": False})
    df["resolution_hours"] = (df["completed_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["response_hours"] = (df["assigned_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["age_hours_at_extract"] = (extract - df["created_at"]).dt.total_seconds() / 3600

    unusable = (df["dq_future_created"] | df["dq_date_order"] | df["dq_completed_no_date"]
                | df["dq_open_with_end"] | df["created_at"].isna() | df["priority"].isna())
    ls = pd.Series("excluded_dq", index=df.index)
    closed = (df["status"] == "Completed") & ~unusable
    ls[closed] = "labeled_closed"
    is_open = df["status"].isin(["In Progress", "Waiting"]) & ~unusable
    ls[is_open & (df["age_hours_at_extract"] > df["completion_sla_h"])] = "labeled_open_breached"
    ls[is_open & (df["age_hours_at_extract"] <= df["completion_sla_h"])] = "censored_open"
    ls[(df["status"] == "Cancelled") & ~unusable] = "cancelled"
    df["label_status"] = ls
    df["sla_breach"] = np.nan
    df.loc[ls == "labeled_closed", "sla_breach"] = (
        df.loc[ls == "labeled_closed", "resolution_hours"] > df.loc[ls == "labeled_closed", "completion_sla_h"]).astype(int)
    df.loc[ls == "labeled_open_breached", "sla_breach"] = 1
    for k, v in ls.value_counts().items():
        log("8 label", f"label_status = {k}", v, "see rule",
            {"labeled_closed": "closed with valid dates: breach = resolution_hours > completion SLA",
             "labeled_open_breached": "still open and already older than SLA: breach is already certain",
             "censored_open": "still open and within SLA: outcome unknown - NOT labelled 0",
             "cancelled": "excluded from SLA performance", "excluded_dq": "failed a validity rule"}[k])

    df = df.sort_values("created_at").reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "work_orders_clean.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(LOG).to_csv(os.path.join(a.out, "cleaning_log.csv"), index=False, encoding="utf-8-sig")

    lab = df[df["sla_breach"].notna()]
    print(f"raw rows {n0} -> clean rows {len(df)}")
    print(df["label_status"].value_counts().to_string())
    print(f"labelled rows {len(lab)} | breach rate {lab['sla_breach'].mean():.3f}")
    print(pd.DataFrame(LOG)[["step", "issue", "rows_affected"]].to_string(index=False))


if __name__ == "__main__":
    main()
