"""
analytics.py - every number shown in the app is computed here (no LLM, no invented values).
Messages are bilingual through i18n.tr(arabic, english).
"""
import json
import re

import numpy as np
import pandas as pd

from build_features import DATA, EXTRACT, FEATURES, add_features
from i18n import tr
from train_models import boosting

PRIORITIES = ["P1-Emergency", "P2-High", "P3-Medium", "P4-Low"]
SCOPE_GROUP = {"HVAC": "MEP", "Electrical": "MEP", "Plumbing": "MEP",
               "Civil": "Civil", "Fire Safety": "Safety", "Elevators": "Safety"}
DRIVERS = {"breach_rate": ("نسبة خرق SLA", "SLA breach rate"),
           "high_priority_breach_rate": ("خرق الأولويات العالية", "High-priority breaches"),
           "overdue_severity": ("شدة التأخير", "Overdue severity"),
           "avg_backlog": ("تراكم الأعمال", "Backlog"),
           "repeat_rate": ("تكرار الأعطال", "Repeat faults")}


def driver_names(text):
    return [tr(*DRIVERS[x]) for x in text.split(" + ")]


def load_all():
    wo = pd.read_csv(f"{DATA}/work_orders_clean.csv", encoding="utf-8-sig",
                     parse_dates=["created_at", "completed_at"])
    fac = pd.read_csv(f"{DATA}/facilities.csv")
    read = lambda n: pd.read_csv(f"{DATA}/{n}")
    with open(f"{DATA}/test_summary.json", encoding="utf-8") as f:
        test = json.load(f)
    return {"source": "synthetic", "asof": EXTRACT, "sla": read("sla_matrix.csv"), "periods": None,
            "note": None, "wo": wo, "fac": fac, "features": read("model_features.csv"),
            "risk_f": read("risk_facility.csv"), "risk_c": read("risk_contract.csv"),
            "results": read("results_test.csv"), "importance": read("feature_importance.csv"),
            "calibration": read("calibration_test.csv"), "groups": read("group_check_contractor.csv"),
            "risk_val": read("risk_validation.csv"), "test": test}


def train_model(features):
    tr_rows = features[features.split == "train"]
    return boosting(2, 0.05).fit(tr_rows[FEATURES], tr_rows.sla_breach.astype(int))


# ------------------------------------------------------------------ KPIs
def valid_rows(wo):
    return wo[~wo.dq_future_created & ~wo.dq_date_order & wo.created_at.notna()]


def period(wo, start, end):
    v = valid_rows(wo)
    return v[(v.created_at >= start) & (v.created_at < end)]


def kpis(d):
    known = d[d.sla_breach.notna()]
    return {"total": len(d), "completed": int((d.status == "Completed").sum()),
            "open": int(d.status.isin(["In Progress", "Waiting"]).sum()),
            "cancelled": int((d.status == "Cancelled").sum()), "known": len(known),
            "breaches": int(known.sla_breach.sum()),
            "compliance": float(1 - known.sla_breach.mean()) if len(known) else np.nan}


def by_group(d, key):
    d = d[d.status != "Cancelled"].copy()
    d["ratio"] = d.resolution_hours / d.completion_sla_h
    d["repeat"] = d.duplicated(["asset_id", "scope"], keep="first") & d.asset_id.notna()
    g = d.groupby(key).agg(work_orders=("wo_id", "size"), known=("sla_breach", "count"),
                           breaches=("sla_breach", "sum"), median_time_vs_sla=("ratio", "median"),
                           median_response_h=("response_hours", "median"),
                           repeat_share=("repeat", "mean"))
    g["breach_rate"] = g.breaches / g.known.replace(0, np.nan)
    g["open_now"] = d[d.status.isin(["In Progress", "Waiting"])].groupby(key).size()
    g["open_now"] = g.open_now.fillna(0).astype(int)
    return g.sort_values("breach_rate", ascending=False)


def monthly(d, key=None):
    d = d[d.sla_breach.notna()].assign(month=lambda x: x.created_at.dt.to_period("M").dt.to_timestamp())
    if key:
        return d.groupby(["month", key]).sla_breach.mean().unstack()
    return d.groupby("month").sla_breach.agg(work_orders="size", breach_rate="mean")


def compare(wo, days=90, key="region", end=EXTRACT):
    mid, start = end - pd.Timedelta(days=days), end - pd.Timedelta(days=2 * days)
    a, b = by_group(period(wo, start, mid), key), by_group(period(wo, mid, end), key)
    t = pd.DataFrame({"previous_rate": a.breach_rate, "recent_rate": b.breach_rate,
                      "previous_n": a.known, "recent_n": b.known})
    t["change"] = t.recent_rate - t.previous_rate
    # approximate two-proportion z score: is the change larger than random variation?
    p = (t.previous_rate * t.previous_n + t.recent_rate * t.recent_n) / (t.previous_n + t.recent_n)
    se = np.sqrt(p * (1 - p) * (1 / t.previous_n + 1 / t.recent_n))
    t["beyond_noise"] = (t.change.abs() / se > 1.96).map({True: tr("نعم", "Yes"), False: tr("لا", "No")})
    return t.sort_values("change", ascending=False)


def recurring(wo, days=90, top=10, end=EXTRACT):
    d = period(wo, end - pd.Timedelta(days=days), end)
    d = d[(d.status != "Cancelled") & d.asset_id.notna()]
    g = d.groupby(["asset_id", "scope", "facility_id"]).agg(
        reports=("wo_id", "size"), breaches=("sla_breach", "sum"))
    return g[g.reports >= 3].sort_values(["reports", "breaches"], ascending=False).head(top).reset_index()


def associations(d):
    """Breach rate under each condition. Associations only, not causes."""
    k = d[d.sla_breach.notna()].copy()
    k["after_hours"] = ((k.created_at.dt.hour >= 18) | (k.created_at.dt.hour < 6)
                        | k.created_at.dt.day_of_week.isin([4, 5]))
    overall = k.sla_breach.mean()
    rows = []
    for col, label in [("after_hours", tr("خارج الدوام", "After hours")),
                       ("contractor_id", tr("المقاول", "Contractor")), ("region", tr("المنطقة", "Region")),
                       ("scope", tr("نطاق العمل", "Scope")), ("priority", tr("الأولوية", "Priority"))]:
        for val, g in k.groupby(col):
            if len(g) >= 50 and val is not False:
                name = label if col == "after_hours" else f"{label}: {val}"
                rows.append({"factor": name, "work_orders": len(g), "breach_rate": g.sla_breach.mean(),
                             "difference_vs_average": g.sla_breach.mean() - overall})
    return pd.DataFrame(rows).sort_values("difference_vs_average", ascending=False), overall


# ------------------------------------------------------------------ prediction
OPEN_COLUMNS = ["wo_id", "priority", "scope", "region", "facility_id", "contractor_id", "probability",
                "alert", "hours_left", "contractor_backlog", "facility_breach_180d", "asset_repeat_30d"]


def open_orders(data, model):
    wo = data["wo"]
    ids = wo[wo.label_status == "censored_open"].wo_id
    f = data["features"]
    o = f[f.wo_id.isin(ids)].copy()
    if model is None or o.empty:
        return pd.DataFrame(columns=OPEN_COLUMNS).astype({"alert": bool})
    o["probability"] = model.predict_proba(o[FEATURES])[:, 1]
    o["alert"] = o.probability >= data["test"]["threshold"]
    o = o.merge(wo[["wo_id", "completion_sla_h", "age_hours_at_extract"]], on="wo_id")
    o["hours_left"] = o.completion_sla_h - o.age_hours_at_extract
    return o.sort_values("probability", ascending=False)


REQUIRED = ["wo_id", "created_at", "priority", "scope", "facility_id"]


def validate_upload(up, data):
    """Return (valid rows, rejected rows with reasons, fatal message). Never guesses a missing value."""
    up = up.copy()
    up.columns = [str(c).strip().lower() for c in up.columns]
    missing = [c for c in REQUIRED if c not in up.columns]
    if missing:
        return None, None, tr("الملف تنقصه أعمدة إلزامية: ", "Required columns are missing: ") + ", ".join(missing)
    if up.empty:
        return None, None, tr("الملف لا يحتوي على صفوف.", "The file has no rows.")
    if "asset_id" not in up.columns:
        up["asset_id"] = np.nan
    for c in ["wo_id", "priority", "scope", "facility_id", "asset_id"]:
        up[c] = up[c].astype("string").str.strip()
    dates = pd.to_datetime(up.created_at, format="%Y-%m-%d %H:%M", errors="coerce")
    known_fac, known_ids = set(data["fac"].facility_id), set(data["wo"].wo_id)
    cut = data["asof"] - pd.Timedelta(days=7)
    scopes = set(data["wo"].scope.dropna())
    reasons = []
    dup = up.wo_id.duplicated(keep=False)
    for i, r in up.iterrows():
        why = []
        for c in REQUIRED:
            if pd.isna(r[c]) or str(r[c]) == "":
                why.append(tr(f"{c} فارغ", f"{c} is empty"))
        if pd.notna(r.created_at) and pd.isna(dates[i]):
            why.append(tr("صيغة التاريخ يجب أن تكون YYYY-MM-DD HH:MM", "Date must be YYYY-MM-DD HH:MM"))
        if pd.notna(r.priority) and r.priority not in PRIORITIES:
            why.append(tr("أولوية غير معروفة: ", "Unknown priority: ") + r.priority)
        if pd.notna(r.scope) and r.scope not in scopes:
            why.append(tr("نطاق غير معروف: ", "Unknown scope: ") + r.scope)
        if pd.notna(r.facility_id) and r.facility_id not in known_fac:
            why.append(tr("مرفق غير موجود: ", "Facility not found: ") + r.facility_id)
        if pd.notna(r.wo_id) and dup[i]:
            why.append(tr("رقم أمر العمل مكرر في الملف", "Work order ID repeated in the file"))
        if pd.notna(r.wo_id) and r.wo_id in known_ids:
            why.append(tr("رقم أمر العمل موجود في السجل التاريخي", "Work order ID already exists in history"))
        if pd.notna(dates[i]) and dates[i] < cut:
            why.append(tr(f"تاريخ أقدم من فترة التنبؤ (قبل {cut:%Y-%m-%d})",
                          f"Date is older than the prediction window (before {cut:%Y-%m-%d})"))
        if pd.notna(dates[i]) and dates[i] > data["asof"] + pd.Timedelta(days=30):
            why.append(tr("تاريخ أبعد من 30 يوماً بعد آخر تحديث للبيانات",
                          "Date is more than 30 days after the last data update"))
        reasons.append(tr("، ", "; ").join(why))
    up["rejection_reason"] = reasons
    bad = up[up.rejection_reason != ""]
    good = up[up.rejection_reason == ""].drop(columns="rejection_reason")
    good = good.assign(created_at=dates[good.index])
    return good, bad, None


def score_new(good, data, model):
    wo, fac = data["wo"], data["fac"]
    sla = data["sla"]
    new = good[["wo_id", "created_at", "priority", "scope", "facility_id", "asset_id"]].copy()
    new = new.merge(fac[["facility_id", "region", "contract_id"]], on="facility_id", how="left")
    # contractor = the one that usually serves this contract and scope in the history
    top = lambda s: s.mode().iloc[0]
    exact = wo.groupby(["contract_id", "scope"]).contractor_id.agg(top).to_dict()
    by_contract = wo.groupby("contract_id").contractor_id.agg(top).to_dict()
    new["contractor_id"] = [exact.get((c, s), by_contract.get(c)) for c, s in zip(new.contract_id, new.scope)]
    new = new.merge(sla, on="priority", how="left")
    new["status"], new["completed_at"], new["sla_breach"] = "Waiting", pd.NaT, np.nan
    flags = ["dq_future_created", "dq_date_order", "dq_completed_no_date"]
    for c in flags:
        new[c] = False
    both = pd.concat([wo, new], ignore_index=True)
    for c in flags:
        both[c] = both[c].astype(bool)
    f = add_features(both, fac)
    f = f[f.wo_id.isin(new.wo_id)].copy()
    f["probability"] = model.predict_proba(f[FEATURES])[:, 1]
    f["alert"] = f.probability >= data["test"]["threshold"]
    return f.sort_values("probability", ascending=False)


# ------------------------------------------------------------------ recommendations
def recommendations(data, model, days=90):
    """priority is a code (urgent / high / medium); ids lists the work orders concerned."""
    wo, risk, asof = data["wo"], data["risk_f"], data["asof"]
    recs = []

    def add(priority, action, evidence, owner, ids=()):
        recs.append({"priority": priority, "action": action, "evidence": evidence, "owner": owner,
                     "ids": list(ids)})

    o = open_orders(data, model)
    n = int(o.alert.sum())
    if n:
        add("urgent", tr("متابعة أوامر العمل المفتوحة ذات التنبيه", "Follow up the flagged open work orders"),
            tr(f"{n} أمر مفتوح فوق عتبة التنبيه من أصل {len(o)}",
               f"{n} of {len(o)} open work orders are above the alert threshold"),
            tr("مهندس التخطيط", "Planning engineer"), o[o.alert].head(5).wo_id)
    for _, r in risk.head(3).iterrows():
        d = tr(" و", " and ").join(driver_names(r.main_drivers))
        add("high", tr(f"مراجعة المرفق {r.facility_id} مع مشرف الموقع",
                       f"Review facility {r.facility_id} with the site supervisor"),
            tr(f"درجة المخاطر {r.risk_score:.0f}/100؛ نسبة خرق مرجّحة {r.breach_rate:.0%} من "
               f"{int(r.work_orders)} أمر؛ أبرز العوامل: {d}",
               f"Risk score {r.risk_score:.0f}/100; smoothed breach rate {r.breach_rate:.0%} over "
               f"{int(r.work_orders)} work orders; main drivers: {d}"),
            tr("مدير المرفق", "Facility manager"))
    c = by_group(period(wo, asof - pd.Timedelta(days=days), asof), "contractor_id")
    avg = c.breaches.sum() / c.known.sum()
    for cid, r in c[(c.breach_rate > avg * 1.2) & (c.known >= 50)].iterrows():
        add("high", tr(f"اجتماع أداء مع {cid} وخطة تصحيحية", f"Performance meeting with {cid} and a corrective plan"),
            tr(f"نسبة الخرق {r.breach_rate:.0%} مقابل متوسط {avg:.0%} خلال {days} يوماً "
               f"({int(r.breaches)} من {int(r.known)})؛ يلزم فحص نوع الأعمال قبل الحكم",
               f"Breach rate {r.breach_rate:.0%} vs {avg:.0%} average over {days} days "
               f"({int(r.breaches)} of {int(r.known)}); check the work mix before judging"),
            tr("مدير العقد", "Contract manager"))
    for _, r in recurring(wo, days, 3, asof).iterrows():
        add("medium", tr(f"تحليل سبب جذري للأصل {r.asset_id} ({r.scope})",
                         f"Root cause analysis for asset {r.asset_id} ({r.scope})"),
            tr(f"{int(r.reports)} بلاغات خلال {days} يوماً", f"{int(r.reports)} reports in {days} days"),
            tr("المهندس الفني", "Technical engineer"))
    return pd.DataFrame(recs)


# ------------------------------------------------------------------ rule-based copilot
def normalise(q):
    q = q.lower().strip()
    q = re.sub(r"[ً-ْ]", "", q)
    return q.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")


OUT_OF_DATA = ["تكلف", "غرام", "ميزاني", "ايراد", "ربح", "راتب", "عماله", "موظف", "رضا", "شكاو",
               "طقس", "cost", "budget", "penalt", "salary", "revenue", "manpower", "satisfaction", "weather"]

SUGGESTIONS = [("ما أكثر العقود خطراً؟", "Which contracts carry the highest risk?"),
               ("ما المرافق التي تحتاج تدخلاً فورياً؟", "Which facilities need immediate intervention?"),
               ("أي أوامر العمل المفتوحة مرجّح تأخرها؟", "Which open work orders are likely to be late?"),
               ("ما المقاولون الأعلى تأخيراً؟", "Which contractors have the highest delay rates?"),
               ("ما العوامل المرتبطة بخرق SLA؟", "What factors are linked to SLA breaches?"),
               ("أي المناطق يتدهور أداؤها؟", "Which regions are deteriorating?"),
               ("قارن آخر 90 يوماً بالفترة السابقة", "Compare the last 90 days with the previous period"),
               ("ما أنماط الأعطال المتكررة؟", "What are the recurring failure patterns?"),
               ("بماذا توصي؟", "What do you recommend?")]


def suggestions():
    return [tr(*s) for s in SUGGESTIONS]


def copilot(question, data, model):
    """Intent matching by keywords. Returns (answer, table or None, source)."""
    q = normalise(question)
    has = lambda *w: any(normalise(x) in q for x in w)
    wo, asof = data["wo"], data["asof"]
    m = re.search(r"(\d{2,3})\s*(يوم|day)", q)
    days = int(m.group(1)) if m and 14 <= int(m.group(1)) <= 360 else 90
    recent = period(wo, asof - pd.Timedelta(days=days), asof)

    if any(w in q for w in OUT_OF_DATA):
        return (tr("لا أستطيع الإجابة: البيانات المتاحة لا تحتوي على هذه المعلومة "
                   "(لا توجد تكاليف أو غرامات أو موارد بشرية أو رضا عملاء). لن أقدّر رقماً من عندي.",
                   "I cannot answer: the data does not contain this information (no costs, penalties, "
                   "manpower or satisfaction data). I will not make up a number."), None, "—")
    if has("توصي", "توصيه", "اجراء", "تحسين", "recommend", "action"):
        return (tr(f"الإجراءات المقترحة بناءً على آخر {days} يوماً (تحتاج مراجعة بشرية):",
                   f"Suggested actions based on the last {days} days (human review required):"),
                recommendations(data, model, days), "analytics.recommendations")
    if has("مفتوح", "مرجح", "تنبؤ", "open", "likely", "predict"):
        o = open_orders(data, model)
        if model is None:
            return (tr("التنبؤ غير متاح لهذه البيانات لأنها لا تكفي لتدريب نموذج.",
                       "Prediction is unavailable: this data is too small to train a model."), None, "—")
        return (tr(f"{int(o.alert.sum())} من {len(o)} أمر عمل مفتوح فوق عتبة التنبيه. الاحتمال تقديري وليس مؤكداً.",
                   f"{int(o.alert.sum())} of {len(o)} open work orders are above the alert threshold. "
                   "Probabilities are estimates."),
                o[["wo_id", "priority", "facility_id", "contractor_id", "probability", "hours_left", "alert"]]
                .head(10), "analytics.open_orders")
    if has("عقد", "عقود") or re.search(r"\bcontracts?\b", q):
        t = data["risk_c"][["contract_id", "work_orders", "breach_rate", "risk_score", "main_drivers"]]
        r = t.iloc[0]
        return (tr(f"أعلى عقد خطراً: {r.contract_id} بدرجة {r.risk_score:.0f}/100 ونسبة خرق {r.breach_rate:.0%} "
                   "خلال آخر 90 يوماً. الدرجة قاعدية وليست تنبؤاً.",
                   f"Highest-risk contract: {r.contract_id}, score {r.risk_score:.0f}/100, breach rate "
                   f"{r.breach_rate:.0%} over the last 90 days. The score is rule-based, not a prediction."),
                t, "risk_contract.csv")
    if has("مقاول", "contractor", "vendor"):
        t = by_group(recent, "contractor_id")[["work_orders", "breaches", "breach_rate", "median_time_vs_sla"]]
        return (tr(f"أعلى نسبة خرق خلال {days} يوماً: {t.index[0]} ({t.breach_rate.iloc[0]:.0%}). "
                   "لا يثبت هذا مسؤولية المقاول وحده؛ نوع الأعمال والمرافق يختلف.",
                   f"Highest breach rate over {days} days: {t.index[0]} ({t.breach_rate.iloc[0]:.0%}). "
                   "This alone does not prove contractor fault; work mix and facilities differ."),
                t.reset_index(), "analytics.by_group")
    if has("سبب", "اسباب", "عوامل", "لماذا", "why", "cause", "reason", "factor"):
        t, overall = associations(recent)
        return (tr(f"عوامل ترتبط بارتفاع الخرق مقارنة بالمتوسط ({overall:.0%}) خلال {days} يوماً. "
                   "هذه ارتباطات لا تثبت السببية.",
                   f"Factors associated with more breaches than average ({overall:.0%}) over {days} days. "
                   "These are associations, not proven causes."), t.head(8), "analytics.associations")
    if has("متكرر", "تكرار", "recurr", "repeat", "pattern", "انماط"):
        return (tr(f"أصول عليها 3 بلاغات أو أكثر خلال {days} يوماً. السبب الجذري يحتاج فحصاً ميدانياً.",
                   f"Assets with 3 or more reports in {days} days. Root cause needs a site inspection."),
                recurring(wo, days, 10, asof), "analytics.recurring")
    if has("منطق", "مناطق", "region", "تدهور", "deteriorat"):
        t = compare(wo, days, "region", asof)
        w = ", ".join(t[(t.change > 0) & (t.beyond_noise == tr("نعم", "Yes"))].index)
        msg = (tr(f"مناطق ارتفع فيها الخرق بأكثر من التذبذب العشوائي: {w}.",
                  f"Regions where breaches rose beyond random variation: {w}.") if w
               else tr("لا توجد منطقة ارتفع فيها الخرق بما يتجاوز التذبذب العشوائي.",
                       "No region rose beyond random variation."))
        return (msg + tr(f" (آخر {days} يوماً مقابل ما قبلها)", f" (last {days} days vs the period before)"),
                t.reset_index(), "analytics.compare")
    if has("قارن", "مقارنه", "compare", "فتره", "period"):
        a = kpis(period(wo, asof - pd.Timedelta(days=2 * days), asof - pd.Timedelta(days=days)))
        b = kpis(recent)
        t = pd.DataFrame({tr("السابقة", "Previous"): a, tr("الحالية", "Current"): b})
        return (tr(f"الالتزام بـ SLA: {a['compliance']:.1%} ثم {b['compliance']:.1%} (فترتان من {days} يوماً).",
                   f"SLA compliance: {a['compliance']:.1%} then {b['compliance']:.1%} (two periods of {days} days)."),
                t.reset_index(names=tr("المؤشر", "KPI")), "analytics.kpis")
    if has("مرفق", "مرافق", "facilit", "تدخل", "موقع", "intervention", "site"):
        t = data["risk_f"].head(8)[["facility_id", "region", "risk_score", "risk_class",
                                    "breach_rate", "work_orders", "main_drivers"]]
        return (tr("المرافق الأعلى أولوية للتدخل حسب درجة المخاطر (آخر 90 يوماً):",
                   "Facilities with the highest intervention priority by risk score (last 90 days):"),
                t, "risk_facility.csv")
    return (tr("لم أفهم السؤال أو أنه خارج نطاقي. جرّب أحد الأسئلة المقترحة.",
               "I did not understand, or it is outside my scope. Try one of the suggested questions."), None, "—")
