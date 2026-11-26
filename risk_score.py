"""
risk_score.py - transparent rule-based risk score for facilities and contracts (NOT the ML model).

Usage:   python risk_score.py
Requirements: pip install pandas numpy
"""
import json

import numpy as np
import pandas as pd

from build_features import DATA, EXTRACT, add_features, load

WINDOW_DAYS, SMOOTH = 90, 20
INDICATORS = ["breach_rate", "high_priority_breach_rate", "overdue_severity", "avg_backlog", "repeat_rate"]
EQUAL = np.repeat(0.2, 5)
PRIORITY_WEIGHTED = np.array([0.35, 0.25, 0.15, 0.10, 0.15])  # alternative judgement, for sensitivity


def indicators(df, key, asof):
    """One row per facility/contract, using only what was known before `asof`."""
    w = df[(df.created_at >= asof - pd.Timedelta(days=WINDOW_DAYS)) & (df.created_at < asof)
           & ~df.dq_future_created & ~df.dq_date_order & (df.status != "Cancelled")].copy()
    known = w.outcome_known_at < asof
    w["known"] = known.astype(int)
    w["breach"] = np.where(known, w.sla_breach.fillna(0), 0)
    w["hp_breach"] = w.breach * w.priority.isin(["P1-Emergency", "P2-High"])
    # overdue: hours beyond SLA as a multiple of SLA, capped at 3, for breaches already finished
    ratio = (w.resolution_hours / w.completion_sla_h - 1).clip(0, 3)
    w["overdue"] = np.where(known & (w.breach == 1) & (w.completed_at < asof), ratio, 0)
    w["repeat"] = (w.asset_repeat_30d > 0).astype(int)
    backlog_col = "facility_backlog" if key == "facility_id" else "contractor_backlog"
    g = w.groupby(key).agg(work_orders=("wo_id", "size"), known=("known", "sum"), breaches=("breach", "sum"),
                           hp=("hp_breach", "sum"), overdue=("overdue", "sum"),
                           avg_backlog=(backlog_col, "mean"), repeat_rate=("repeat", "mean"))
    if key == "contract_id":  # contract backlog = sum of its facilities' average backlog
        per_fac = w.groupby([key, "facility_id"]).facility_backlog.mean()
        g["avg_backlog"] = per_fac.groupby(level=0).mean()
    base = g.breaches.sum() / g.known.sum()
    base_hp = g.hp.sum() / g.known.sum()
    g["breach_rate"] = (g.breaches + SMOOTH * base) / (g.known + SMOOTH)
    g["high_priority_breach_rate"] = (g.hp + SMOOTH * base_hp) / (g.known + SMOOTH)
    g["overdue_severity"] = g.overdue / (g.known + SMOOTH)
    return g


def score(g, weights, ref=None):
    ref = g if ref is None else ref
    lo, hi = ref[INDICATORS].min(), ref[INDICATORS].max()
    z = ((g[INDICATORS] - lo) / (hi - lo).replace(0, 1)).clip(0, 1) * 100
    return z @ weights, z * weights


def build(df, key, asof, ref=None):
    g = indicators(df, key, asof)
    g["risk_score"], parts = score(g, EQUAL, ref)
    g["alt_score"], _ = score(g, PRIORITY_WEIGHTED, ref)
    g["main_drivers"] = parts.apply(lambda r: " + ".join(r.sort_values(ascending=False).index[:2]), axis=1)
    return g.sort_values("risk_score", ascending=False)


def main():
    wo, fac = load()
    df = add_features(wo, fac)

    f = build(df, "facility_id", EXTRACT)
    rank = f.risk_score.rank(pct=True)
    f["risk_class"] = np.select([rank > 0.8, rank > 0.5], ["High", "Medium"], "Low")
    f = f.join(fac.set_index("facility_id")[["region", "contract_id", "facility_type"]])
    f.round(4).to_csv(f"{DATA}/risk_facility.csv")

    c = build(df, "contract_id", EXTRACT, ref=indicators(df, "facility_id", EXTRACT))
    c.round(4).to_csv(f"{DATA}/risk_contract.csv")

    # sensitivity to weights
    top = set(f.head(8).index)
    rng = np.random.default_rng(0)
    hits = pd.Series(0.0, index=f.index)
    overlaps = []
    for _ in range(1000):
        s, _ = score(f, rng.dirichlet(np.ones(5) * 4))
        t = s.sort_values(ascending=False).head(8).index
        hits[t] += 1
        overlaps.append(len(top & set(t)) / 8)
    (hits / 1000).rename("share_in_top8").round(3).to_csv(f"{DATA}/risk_sensitivity.csv")

    # forward validation: does the score at time t rank next-quarter breach rates?
    val = []
    for asof in pd.to_datetime(["2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]):
        s = build(df, "facility_id", asof)
        nxt = df[(df.created_at >= asof) & (df.created_at < asof + pd.Timedelta(days=90)) & df.sla_breach.notna()]
        future = nxt.groupby("facility_id").sla_breach.mean()
        j = s.join(future.rename("future"), how="inner")
        top8 = j.nlargest(8, "risk_score").future.mean()
        val.append({"as_of": str(asof.date()),
                    "spearman_score_vs_next": j.risk_score.rank().corr(j.future.rank()),
                    "spearman_past_breach_only": j.breach_rate.rank().corr(j.future.rank()),
                    "next_rate_top8": top8, "next_rate_all": j.future.mean()})
    val = pd.DataFrame(val).round(3)
    val.to_csv(f"{DATA}/risk_validation.csv", index=False)
    summary = {"top8_overlap_equal_vs_priority_weights":
                   len(top & set(f.nlargest(8, "alt_score").index)) / 8,
               "mean_top8_overlap_random_weights": float(np.mean(overlaps)),
               "facilities_in_top8_in_90pct_of_weightings": int((hits >= 900).sum())}
    with open(f"{DATA}/risk_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f[["work_orders", "breach_rate", "risk_score", "risk_class", "main_drivers", "region"]].head(8).round(3).to_string())
    print(c[["work_orders", "breach_rate", "risk_score", "main_drivers"]].round(3).to_string())
    print(val.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
