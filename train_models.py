"""
train_models.py - choose a model on VALIDATION, then evaluate it ONCE on TEST.

Usage:   python train_models.py            (validation comparison only)
         python train_models.py --final    (adds the one-time test evaluation + exports)
Requirements: pip install pandas numpy scikit-learn
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from build_features import DATA, FEATURES
from train_baseline import CAPACITY, CATEGORICAL, NUMERIC, evaluate


def logistic(C=1.0):
    return Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                    ("num", StandardScaler(), NUMERIC)])),
        ("clf", LogisticRegression(C=C, max_iter=3000))])


def boosting(depth, rate):
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                               CATEGORICAL)], remainder="passthrough")
    clf = HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=rate, max_iter=200, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=0,
        categorical_features=list(range(len(CATEGORICAL))))
    return Pipeline([("prep", prep), ("clf", clf)])


def rule_scores(tr, other):
    rule = tr.groupby(["contractor_id", "priority"]).sla_breach.mean().rename("p").reset_index()
    return other.merge(rule, on=["contractor_id", "priority"], how="left").p.fillna(tr.sla_breach.mean()).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(f"{DATA}/model_features.csv")
    tr, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    y = lambda d: d.sla_breach.values.astype(int)

    candidates = {f"LogReg C={c}": logistic(c) for c in (0.03, 0.1, 1.0)}
    candidates.update({f"Boosting depth={d} lr={r}": boosting(d, r) for d in (2, 3) for r in (0.02, 0.05)})
    rows = []
    for name, m in candidates.items():
        m.fit(tr[FEATURES], y(tr))
        rows.append(evaluate(name, y(va), m.predict_proba(va[FEATURES])[:, 1]))
    res = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).round(4)
    res.to_csv(f"{DATA}/results_models_validation.csv", index=False)
    print(res[["model", "PR_AUC", "ROC_AUC", "Brier", "precision_top20", "recall_top20"]].to_string(index=False))
    if not a.final:
        return

    # selection rule: boosting must beat the best logistic model by > 0.01 PR-AUC to justify the complexity
    best = res.iloc[0]
    best_lr = res[res.model.str.startswith("LogReg")].iloc[0]
    chosen = best.model if best.PR_AUC - best_lr.PR_AUC > 0.01 else best_lr.model
    model = candidates[chosen]
    p_va = model.predict_proba(va[FEATURES])[:, 1]
    threshold = float(np.quantile(p_va, 1 - CAPACITY))  # set on validation only
    print(f"\nchosen: {chosen} | alert threshold (top {CAPACITY:.0%} on validation): {threshold:.3f}")

    p_te = model.predict_proba(te[FEATURES])[:, 1]
    yte = y(te)
    final = pd.DataFrame([evaluate("B0 same rate for all", yte, np.full(len(te), y(tr).mean())),
                          evaluate("B1 contractor x priority rate", yte, rule_scores(tr, te)),
                          evaluate(chosen, yte, p_te)]).round(4)
    final.to_csv(f"{DATA}/results_test.csv", index=False)
    print("\nTEST (used once)\n" + final.to_string(index=False))

    flag = p_te >= threshold
    tp, fp = int((flag & (yte == 1)).sum()), int((flag & (yte == 0)).sum())
    fn, tn = int((~flag & (yte == 1)).sum()), int((~flag & (yte == 0)).sum())
    # bootstrap interval for the improvement over the rule
    rng, p_rule = np.random.default_rng(0), rule_scores(tr, te)
    gains = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        gains.append(average_precision_score(yte[i], p_te[i]) - average_precision_score(yte[i], p_rule[i]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    lead = te.assign(f=flag).query("f and sla_breach == 1").priority.map(
        {"P1-Emergency": 4, "P2-High": 24, "P3-Medium": 72, "P4-Low": 168})
    summary = {"chosen_model": chosen, "threshold": threshold, "test_rows": int(len(yte)),
               "test_breaches": int(yte.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
               "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
               "flag_rate": float(flag.mean()),
               "pr_auc_gain_over_rule_95ci": [float(lo), float(hi)],
               "median_warning_window_h_for_caught": float(lead.median())}
    with open(f"{DATA}/test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # probability calibration by band
    cal = pd.DataFrame({"p": p_te, "y": yte})
    cal["band"] = pd.qcut(cal.p, 5, labels=["1 lowest", "2", "3", "4", "5 highest"])
    cal = cal.groupby("band", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    cal.round(3).to_csv(f"{DATA}/calibration_test.csv")
    print(cal.round(3).to_string())

    imp = permutation_importance(model, va[FEATURES], y(va), scoring="average_precision",
                                 n_repeats=20, random_state=0)
    imp = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean,
                        "std": imp.importances_std}).sort_values("importance", ascending=False)
    imp.round(5).to_csv(f"{DATA}/feature_importance.csv", index=False)
    print(imp.head(8).round(4).to_string(index=False))

    # fairness / stability: alert behaviour per contractor
    g = te.assign(flag=flag).groupby("contractor_id").apply(
        lambda d: pd.Series({"n": len(d), "actual_rate": d.sla_breach.mean(), "flag_rate": d.flag.mean(),
                             "recall": d[d.sla_breach == 1].flag.mean(),
                             "precision": d[d.flag].sla_breach.mean()}), include_groups=False)
    g.round(3).to_csv(f"{DATA}/group_check_contractor.csv")
    print(g.round(3).to_string())

    te.assign(probability=p_te.round(4), alert=flag.astype(int))[
        ["wo_id", "created_at", "priority", "scope", "facility_id", "region", "contract_id",
         "contractor_id", "probability", "alert", "sla_breach"]].to_csv(f"{DATA}/predictions_test.csv", index=False)


if __name__ == "__main__":
    main()
