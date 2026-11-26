"""
build_features.py - leakage-safe features, known at work-order creation time.

Usage:   python build_features.py
         python build_features.py --check     (also runs the "no future information" test)
Requirements: pip install pandas numpy
"""
import argparse

import numpy as np
import pandas as pd

DATA = "synthetic_data"
EXTRACT = pd.Timestamp("2026-06-30 12:00")
TRAIN_START = pd.Timestamp("2024-10-01")
VAL_START = pd.Timestamp("2026-01-01")
TEST_START = pd.Timestamp("2026-04-01")
SMOOTH = 20  # pseudo-observations pulling small-sample rates to the overall rate

FEATURES = ["priority", "scope", "region", "facility_id", "contract_id", "contractor_id",
            "facility_type", "area_m2", "building_age_years", "hour", "day_of_week", "month",
            "after_hours", "contractor_backlog", "facility_backlog", "contractor_new_7d",
            "asset_repeat_30d", "asset_missing", "contractor_breach_90d", "facility_breach_180d"]


def count_before(sorted_times, t, side="left"):
    return np.searchsorted(sorted_times, t, side=side)


def add_features(wo, fac):
    df = wo.sort_values(["created_at", "wo_id"]).reset_index(drop=True)
    df = df.merge(fac[["facility_id", "facility_type", "area_m2", "building_age_years"]],
                  on="facility_id", how="left")
    t = df["created_at"].values
    df["hour"] = df.created_at.dt.hour
    df["day_of_week"] = df.created_at.dt.day_of_week
    df["month"] = df.created_at.dt.month
    df["after_hours"] = ((df.hour >= 18) | (df.hour < 6) | df.day_of_week.isin([4, 5])).astype(int)
    df["asset_missing"] = df["asset_id"].isna().astype(int)

    valid_created = ~(df.dq_future_created | df.dq_date_order) & df.created_at.notna()
    # rows whose open/closed history can be trusted
    in_backlog = valid_created & ~df.dq_completed_no_date & (df.status != "Cancelled")
    # time at which a row's breach outcome became known
    deadline = df.created_at + pd.to_timedelta(df.completion_sla_h, unit="h")
    known_at = df.completed_at.where(df.sla_breach != 1, deadline)
    df["outcome_known_at"] = known_at.where(df.sla_breach.notna())

    # backlog = created before t  minus  completed by t
    for key, name in [("contractor_id", "contractor_backlog"), ("facility_id", "facility_backlog")]:
        out = np.zeros(len(df))
        for g, idx in df.groupby(key).indices.items():
            h = df.iloc[idx][in_backlog.iloc[idx]]
            starts = np.sort(h.created_at.values)
            ends = np.sort(h.completed_at.dropna().values)
            out[idx] = count_before(starts, t[idx]) - count_before(ends, t[idx], "right")
        df[name] = out

    df["contractor_new_7d"] = 0
    for g, idx in df.groupby("contractor_id").indices.items():
        starts = np.sort(df.iloc[idx][valid_created.iloc[idx]].created_at.values)
        df.loc[idx, "contractor_new_7d"] = (count_before(starts, t[idx])
                                            - count_before(starts, t[idx] - np.timedelta64(7, "D")))

    df["asset_repeat_30d"] = 0
    for g, idx in df.groupby(["asset_id", "scope"]).indices.items():
        starts = np.sort(df.iloc[idx][valid_created.iloc[idx]].created_at.values)
        df.loc[idx, "asset_repeat_30d"] = (count_before(starts, t[idx])
                                           - count_before(starts, t[idx] - np.timedelta64(30, "D")))

    # historical breach rates using ONLY outcomes known strictly before t
    hist = df[df.outcome_known_at.notna()].sort_values("outcome_known_at")
    gk = hist.outcome_known_at.values
    gc = np.concatenate([[0], np.cumsum(hist.sla_breach.values)])
    n_all = count_before(gk, t)
    overall = np.divide(gc[n_all], n_all, out=np.zeros(len(df)), where=n_all > 0)

    for key, days, name in [("contractor_id", 90, "contractor_breach_90d"),
                            ("facility_id", 180, "facility_breach_180d")]:
        out = np.zeros(len(df))
        for g, idx in df.groupby(key).indices.items():
            h = hist[hist[key] == g]
            k = h.outcome_known_at.values
            c = np.concatenate([[0], np.cumsum(h.sla_breach.values)])
            hi = count_before(k, t[idx])
            lo = count_before(k, t[idx] - np.timedelta64(days, "D"))
            out[idx] = (c[hi] - c[lo] + SMOOTH * overall[idx]) / (hi - lo + SMOOTH)
        df[name] = out
    return df


def assign_split(df):
    mature = df.created_at <= EXTRACT - pd.Timedelta(hours=float(df.completion_sla_h.max()))
    usable = df.sla_breach.notna() & mature
    split = pd.Series("not_used", index=df.index)
    split[usable & (df.created_at >= TRAIN_START) & (df.outcome_known_at < VAL_START)] = "train"
    split[usable & (df.created_at >= VAL_START) & (df.created_at < TEST_START)] = "validation"
    split[usable & (df.created_at >= TEST_START)] = "test"
    return split


def load():
    wo = pd.read_csv(f"{DATA}/work_orders_clean.csv", encoding="utf-8-sig",
                     parse_dates=["created_at", "completed_at"])
    return wo, pd.read_csv(f"{DATA}/facilities.csv")


def leakage_test(wo, fac, full, cut=pd.Timestamp("2025-09-01")):
    """Rebuild features from a snapshot as it looked at `cut`; earlier rows must not change."""
    snap = wo[wo.created_at < cut].copy()
    unknown = snap.completed_at >= cut
    snap.loc[unknown, "completed_at"] = pd.NaT
    late = snap.created_at + pd.to_timedelta(snap.completion_sla_h, unit="h") < cut
    snap.loc[unknown, "sla_breach"] = np.where(late[unknown] & (snap.sla_breach[unknown] == 1), 1, np.nan)
    a = add_features(snap, fac).set_index("wo_id")[FEATURES]
    b = full.set_index("wo_id").loc[a.index, FEATURES]
    diff = [c for c in FEATURES if not (
        np.allclose(a[c], b[c], equal_nan=True) if a[c].dtype.kind in "ifu" else a[c].equals(b[c]))]
    print(f"leakage test on {len(a)} rows before {cut.date()}: "
          f"{'PASS' if not diff else 'FAIL ' + str(diff)}")
    assert not diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    wo, fac = load()
    df = add_features(wo, fac)
    df["split"] = assign_split(df)
    df[["wo_id", "created_at", "split", "sla_breach"] + FEATURES].to_csv(
        f"{DATA}/model_features.csv", index=False)
    print(df.split.value_counts().to_string())
    print(df[df.split != "not_used"].groupby("split").sla_breach.mean().round(3).to_string())
    print(df[["contractor_backlog", "facility_backlog", "asset_repeat_30d",
              "contractor_breach_90d"]].describe().round(2).to_string())
    if args.check:
        leakage_test(wo, fac, df)
