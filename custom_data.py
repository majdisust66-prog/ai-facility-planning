"""
custom_data.py - turns an uploaded work-order history into the same data structure the pages use.
Nothing is written to disk: the result lives only in the visitor's session.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from build_features import FEATURES, add_features
from i18n import tr
from risk_score import build, indicators
from train_baseline import CAPACITY, evaluate
from train_models import boosting, rule_scores

REQUIRED = ["wo_id", "created_at", "completed_at", "priority", "scope", "facility_id", "region", "contractor_id"]
OPTIONAL = ["status", "contract_id", "asset_id", "assigned_at", "sla_hours"]
PRIORITY_MAP = {"1": "P1-Emergency", "2": "P2-High", "3": "P3-Medium", "4": "P4-Low"}
STATUS_MAP = {"completed": "Completed", "closed": "Completed", "comp": "Completed", "close": "Completed",
              "cancelled": "Cancelled", "canceled": "Cancelled", "can": "Cancelled",
              "in progress": "In Progress", "inprg": "In Progress", "open": "In Progress",
              "waiting": "Waiting", "wappr": "Waiting"}
MAX_ROWS, MIN_ROWS = 60000, 50
# below these the model is not trained (practical minimums chosen for this prototype)
MODEL_MIN = {"labelled": 1000, "breaches": 100, "days": 180, "split_breaches": 20}


def validate_history(raw, sla, today=None):
    """Return (accepted rows, rejected rows with reasons, fatal message). Never guesses a value."""
    today = today or pd.Timestamp.now()
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.columns.duplicated().any():
        return None, None, tr("أسماء أعمدة مكررة في الملف.", "The file repeats a column name.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, None, tr("الملف تنقصه أعمدة إلزامية: ", "Required columns are missing: ") + ", ".join(missing)
    if len(df) > MAX_ROWS:
        return None, None, tr(f"الحد الأقصى {MAX_ROWS:,} صف.", f"Maximum {MAX_ROWS:,} rows.")
    df = df.reindex(columns=REQUIRED + OPTIONAL)
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df = df.drop_duplicates().reset_index(drop=True)  # exact copies carry no extra information
    reasons = pd.Series("", index=df.index)

    def reject(mask, message):
        mask = mask.fillna(False).astype(bool)
        reasons.loc[mask] = reasons.loc[mask] + message + tr("، ", "; ")

    for c in REQUIRED:
        if c != "completed_at":
            reject(df[c].isna(), tr(f"{c} فارغ", f"{c} is empty"))
    reject(df.wo_id.notna() & df.wo_id.duplicated(keep=False),
           tr("رقم أمر العمل مكرر ببيانات مختلفة", "Same work order ID with different data"))
    dates = {}
    for c in ["created_at", "completed_at", "assigned_at"]:
        parsed = pd.to_datetime(df[c], format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
        reject(df[c].notna() & parsed.isna(), tr(f"{c}: التاريخ يجب أن يكون YYYY-MM-DD HH:MM",
                                                 f"{c}: date must be YYYY-MM-DD HH:MM"))
        reject(parsed > today, tr(f"{c} في المستقبل", f"{c} is in the future"))
        dates[c] = parsed
    reject(dates["completed_at"] < dates["created_at"], tr("الإكمال قبل الإنشاء", "Completed before created"))

    code = df.priority.str.extract(r"^[pP]\s*([1-4])", expand=False)
    priority = code.map(PRIORITY_MAP)
    reject(df.priority.notna() & priority.isna(), tr("أولوية غير معروفة (استخدم P1 إلى P4)",
                                                     "Unknown priority (use P1 to P4)"))
    lowered = df.status.str.lower()
    status = lowered.map(STATUS_MAP)
    reject(df.status.notna() & status.isna(), tr("حالة غير معروفة", "Unknown status"))
    derived = pd.Series(np.where(dates["completed_at"].notna(), "Completed", "In Progress"), index=df.index)
    status = status.where(df.status.notna(), derived)
    reject((status == "Completed") & df.completed_at.isna(),
           tr("مكتمل بدون تاريخ إكمال", "Completed without a completion date"))
    reject(status.isin(["In Progress", "Waiting"]) & df.completed_at.notna(),
           tr("مفتوح وله تاريخ إكمال", "Open but has a completion date"))
    hours = pd.to_numeric(df.sla_hours, errors="coerce")
    reject(df.sla_hours.notna() & ~(hours > 0), tr("sla_hours يجب أن يكون رقماً موجباً",
                                                   "sla_hours must be a positive number"))
    default_hours = priority.map(sla.set_index("priority").completion_sla_h)

    bad = df.loc[reasons != "", ["wo_id"]].assign(rejection_reason=reasons[reasons != ""].str.rstrip("،; "))
    ok = reasons == ""
    good = df.loc[ok, ["wo_id", "scope", "facility_id", "region", "contractor_id", "contract_id", "asset_id"]]
    good = good.astype(object).where(good.notna(), np.nan)
    good["contract_id"] = good.contract_id.fillna("N/A")
    good["priority"], good["status"] = priority[ok].astype(object), status[ok].astype(object)
    for c in dates:
        good[c] = dates[c][ok]
    good["completion_sla_h"] = hours.where(hours > 0, default_hours)[ok].astype(float)
    if len(good) < MIN_ROWS:
        return good, bad, tr(f"الصفوف المقبولة {len(good)} فقط؛ يلزم {MIN_ROWS} على الأقل.",
                             f"Only {len(good)} rows accepted; at least {MIN_ROWS} are needed.")
    return good, bad, None


def _work_orders(good):
    wo = good.sort_values(["created_at", "wo_id"]).reset_index(drop=True)
    asof = max(wo.created_at.max(), wo.completed_at.max() if wo.completed_at.notna().any() else wo.created_at.max())
    wo["wo_type"], wo["reopened"] = "Corrective", False
    for flag in ["dq_future_created", "dq_date_order", "dq_completed_no_date"]:
        wo[flag] = False
    wo["resolution_hours"] = (wo.completed_at - wo.created_at).dt.total_seconds() / 3600
    wo["response_hours"] = (wo.assigned_at - wo.created_at).dt.total_seconds() / 3600
    wo["age_hours_at_extract"] = (asof - wo.created_at).dt.total_seconds() / 3600
    closed = wo.status == "Completed"
    is_open = wo.status.isin(["In Progress", "Waiting"])
    late = is_open & (wo.age_hours_at_extract > wo.completion_sla_h)
    wo["label_status"] = np.select([closed, late, is_open], ["labeled_closed", "labeled_open_breached",
                                                             "censored_open"], "cancelled")
    wo["sla_breach"] = np.nan
    wo.loc[closed, "sla_breach"] = (wo.resolution_hours > wo.completion_sla_h)[closed].astype(float)
    wo.loc[late, "sla_breach"] = 1.0
    return wo, asof


def _facilities(wo):
    top = lambda s: s.mode().iloc[0]
    fac = wo.groupby("facility_id").agg(region=("region", top), contract_id=("contract_id", top)).reset_index()
    # type, area and age are not in the upload: a constant tells the model nothing (it cannot bin an empty column)
    fac["facility_type"], fac["area_m2"], fac["building_age_years"] = "—", 0.0, 0.0
    return fac


def _split(df, asof):
    """Time-ordered 60 / 20 / 20 split; training outcomes must be known before validation starts."""
    margin = pd.Timedelta(hours=float(min(df.completion_sla_h.max(), 720)))
    usable = df.sla_breach.notna() & (df.created_at <= asof - margin)
    split = pd.Series("not_used", index=df.index)
    if usable.sum() < 10:
        return split, usable, None
    t_val, t_test = df.created_at[usable].quantile(0.6), df.created_at[usable].quantile(0.8)
    warm = df.created_at.min() + pd.Timedelta(days=30)
    split[usable & (df.created_at >= warm) & (df.created_at < t_val) & (df.outcome_known_at < t_val)] = "train"
    split[usable & (df.created_at >= t_val) & (df.created_at < t_test)] = "validation"
    split[usable & (df.created_at >= t_test)] = "test"
    return split, usable, (warm, t_val, t_test)


def _model(df, usable, cuts):
    """Train and evaluate on the uploaded history, or explain why that is not possible."""
    breaches = int(df.sla_breach[usable].sum())
    days = (df.created_at.max() - df.created_at.min()).days
    parts = {s: df[df.split == s] for s in ("train", "validation", "test")}
    enough = (usable.sum() >= MODEL_MIN["labelled"] and breaches >= MODEL_MIN["breaches"]
              and days >= MODEL_MIN["days"] and cuts is not None
              and all(p.sla_breach.sum() >= MODEL_MIN["split_breaches"]
                      and (p.sla_breach == 0).sum() >= MODEL_MIN["split_breaches"] for p in parts.values()))
    if not enough:
        note = (f"البيانات لا تكفي لتدريب نموذج تنبؤ موثوق: {int(usable.sum()):,} أمر بنتيجة معروفة، "
                f"{breaches} خرقاً، على {days} يوماً. الحد الأدنى {MODEL_MIN['labelled']:,} أمر و"
                f"{MODEL_MIN['breaches']} خرق و{MODEL_MIN['days']} يوماً، مع خروقات في كل فترة تقييم.",
                f"Not enough data to train a trustworthy model: {int(usable.sum()):,} orders with a known outcome, "
                f"{breaches} breaches, over {days} days. Minimum: {MODEL_MIN['labelled']:,} orders, "
                f"{MODEL_MIN['breaches']} breaches and {MODEL_MIN['days']} days, with breaches in every period.")
        empty = pd.DataFrame(columns=["contractor_id", "n", "actual_rate", "flag_rate", "recall", "precision"])
        return None, {"test": None, "results": None, "importance": None, "calibration": None, "groups": empty,
                      "note": note}
    tr_rows, va, te = parts["train"], parts["validation"], parts["test"]
    y = lambda d: d.sla_breach.to_numpy().astype(int)
    model = boosting(2, 0.05).fit(tr_rows[FEATURES], y(tr_rows))
    p_va, p_te, yte = model.predict_proba(va[FEATURES])[:, 1], model.predict_proba(te[FEATURES])[:, 1], y(te)
    threshold = float(np.quantile(p_va, 1 - CAPACITY))
    results = pd.DataFrame([
        evaluate("B0 same rate for all", yte, np.full(len(te), y(tr_rows).mean())),
        evaluate("B1 contractor x priority rate", yte, rule_scores(tr_rows, te)),
        evaluate("Gradient Boosting", yte, p_te)]).round(4)
    flag = p_te >= threshold
    tp, fp = int((flag & (yte == 1)).sum()), int((flag & (yte == 0)).sum())
    fn = int((~flag & (yte == 1)).sum())
    test = {"chosen_model": "Gradient Boosting depth=2", "threshold": threshold, "test_rows": int(len(te)),
            "test_breaches": int(yte.sum()), "TP": tp, "FP": fp, "FN": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "flag_rate": float(flag.mean())}
    cal = pd.DataFrame({"p": p_te, "y": yte})
    cal["band"] = pd.qcut(cal.p.rank(method="first"), 5, labels=["1", "2", "3", "4", "5"]).astype(str)
    cal = cal.groupby("band").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    sample = va.sample(min(len(va), 3000), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], y(sample), scoring="average_precision",
                                 n_repeats=5, random_state=0)
    imp = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean,
                        "std": imp.importances_std}).sort_values("importance", ascending=False)
    rows = []
    for cid, d in te.assign(flag=flag).groupby("contractor_id"):
        rows.append({"contractor_id": cid, "n": len(d), "actual_rate": d.sla_breach.mean(),
                     "flag_rate": d.flag.mean(), "recall": d.flag[d.sla_breach == 1].mean(),
                     "precision": d.sla_breach[d.flag].mean()})
    return model, {"test": test, "results": results, "importance": imp, "calibration": cal,
                   "groups": pd.DataFrame(rows), "note": None}


def _risk(df, fac, asof):
    end = asof + pd.Timedelta(minutes=1)
    f = build(df, "facility_id", end)
    rank = f.risk_score.rank(pct=True)
    f["risk_class"] = np.select([rank > 0.8, rank > 0.5], ["High", "Medium"], "Low")
    f = f.join(fac.set_index("facility_id")[["region", "contract_id", "facility_type"]]).reset_index()
    c = build(df, "contract_id", end, ref=indicators(df, "facility_id", end)).reset_index()
    rows = []
    for k in range(5, 0, -1):  # does the score at time t rank the following 90 days?
        t = (asof - pd.Timedelta(days=90 * k)).normalize()
        if t - pd.Timedelta(days=90) < df.created_at.min():
            continue
        s = build(df, "facility_id", t)
        nxt = df[(df.created_at >= t) & (df.created_at < t + pd.Timedelta(days=90)) & df.sla_breach.notna()]
        j = s.join(nxt.groupby("facility_id").sla_breach.mean().rename("future"), how="inner")
        j = j.dropna(subset=["risk_score", "breach_rate", "future"])
        if len(j) < 5:
            continue
        rows.append({"as_of": str(t.date()),
                     "spearman_score_vs_next": j.risk_score.rank().corr(j.future.rank()),
                     "spearman_past_breach_only": j.breach_rate.rank().corr(j.future.rank()),
                     "next_rate_top8": j.nlargest(8, "risk_score").future.mean(),
                     "next_rate_all": j.future.mean()})
    return f, c, pd.DataFrame(rows).round(3)


def build_dataset(good, sla):
    """Return (data, model). model is None when the history is too small to train on."""
    wo, asof = _work_orders(good)
    if wo.sla_breach.notna().sum() == 0:
        raise ValueError(tr("لا توجد أوامر بنتيجة SLA معروفة.", "No work order has a known SLA outcome."))
    fac = _facilities(wo)
    df = add_features(wo, fac)
    df["split"], usable, cuts = _split(df, asof)
    model, parts = _model(df, usable, cuts)
    risk_f, risk_c, risk_val = _risk(df, fac, asof)
    periods = None
    if model is not None:
        warm, t_val, t_test = cuts
        periods = {"train": f"{warm:%Y-%m-%d} → {t_val:%Y-%m-%d}", "validation": f"{t_val:%Y-%m-%d} → {t_test:%Y-%m-%d}",
                   "test": f"{t_test:%Y-%m-%d} → {asof:%Y-%m-%d}"}
    data = {"source": "custom", "asof": asof + pd.Timedelta(minutes=1), "sla": sla, "wo": wo, "fac": fac,
            "features": df[["wo_id", "created_at", "split", "sla_breach"] + FEATURES],
            "risk_f": risk_f, "risk_c": risk_c, "risk_val": risk_val, "periods": periods, **parts}
    return data, model
