"""
train_baseline.py - baselines on the VALIDATION period. The test period is not touched.

Usage:   python train_baseline.py
Requirements: pip install pandas numpy scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_features import DATA, FEATURES

CATEGORICAL = ["priority", "scope", "region", "facility_id", "contract_id", "contractor_id",
               "facility_type", "month", "day_of_week"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
CAPACITY = 0.20  # share of new work orders the planner can follow up


def evaluate(name, y, p):
    k = int(round(len(y) * CAPACITY))
    # random tie-break so constant / grouped scores are not favoured by row order
    order = np.lexsort((np.random.default_rng(0).random(len(p)), -p))[:k]
    hit = y[order].sum()
    return {"model": name, "PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p), "precision_top20": hit / k, "recall_top20": hit / y.sum(),
            "flagged": k, "caught": int(hit), "false_alarms": int(k - hit),
            "missed": int(y.sum() - hit)}


def main():
    df = pd.read_csv(f"{DATA}/model_features.csv")
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    ytr, yva = tr.sla_breach.values.astype(int), va.sla_breach.values.astype(int)
    results = [evaluate("B0 same rate for all", yva, np.full(len(va), ytr.mean()))]

    keys = ["contractor_id", "priority"]
    rule = tr.groupby(keys).sla_breach.mean().rename("p").reset_index()
    p = va.merge(rule, on=keys, how="left").p.fillna(ytr.mean()).values
    results.append(evaluate("B1 contractor x priority rate", yva, p))

    model = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                    ("num", StandardScaler(), NUMERIC)])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    model.fit(tr[FEATURES], ytr)
    results.append(evaluate("Logistic Regression", yva, model.predict_proba(va[FEATURES])[:, 1]))

    res = pd.DataFrame(results).round(4)
    res.to_csv(f"{DATA}/results_baseline_validation.csv", index=False)
    print(f"train {len(tr)} | validation {len(va)} | validation breach rate {yva.mean():.3f}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
