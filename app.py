"""
app.py - Streamlit prototype, Arabic / English, light / dark.     Run:  python -m streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import analytics as an  # noqa: E402
import custom_data as cd  # noqa: E402
import ui  # noqa: E402
from i18n import set_lang, tr  # noqa: E402

st.set_page_config(page_title="Facility Planning AI", page_icon="assets/logo.svg", layout="wide",
                   initial_sidebar_state="auto")


@st.cache_resource(show_spinner="Loading data and training the model...")
def boot():
    data = an.load_all()
    return data, an.train_model(data["features"])


# language and theme survive a page reload through the URL (?lang=en&theme=dark)
if "theme" not in st.session_state:
    st.session_state.theme = "light" if st.query_params.get("theme") == "light" else "dark"  # dark by default
if "lang" not in st.session_state:
    st.session_state.lang = "en" if st.query_params.get("lang") == "en" else "ar"
top = st.columns(2)
code = top[0].radio("Language", ["ar", "en"], format_func={"ar": "العربية", "en": "English"}.get,
                    horizontal=True, key="lang", label_visibility="collapsed")
set_lang(code)
# After a language switch, send each saved choice to the browser again so its label is redrawn
# in the new language (otherwise the old text stays, or the menu loses its highlight).
if st.session_state.get("shown_lang", code) != code:
    for name in ["theme", "page", "ov_days", "ov_region", "risk_class", "ct_days", "memo_days",
                 "memo_priority", "reviewer", "approve", "demo", "history_source", "consent"]:
        if name in st.session_state:
            st.session_state[name] = st.session_state[name]
st.session_state.shown_lang = code
theme = top[1].radio("Theme", ["light", "dark"], horizontal=True, key="theme", label_visibility="collapsed",
                     format_func=lambda t: tr("☀️ نهاري", "☀️ Light") if t == "light" else tr("🌙 ليلي", "🌙 Dark"))
dark = theme == "dark"
st.query_params["lang"] = code
st.query_params["theme"] = theme
ui.inject(dark)

custom = st.session_state.get("custom")  # an uploaded history, kept for this session only
data, model = (custom["data"], custom["model"]) if custom else boot()
wo, ASOF = data["wo"], data["asof"]
is_custom = data["source"] == "custom"


def reset_choices():
    """Forget selections that may not exist in the other data set."""
    for name in ["ov_region", "memo_priority", "approve", "reviewer", "risk_class", "chat", "demo"]:
        st.session_state.pop(name, None)


def use_sample_data():
    reset_choices()
    st.session_state.pop("custom", None)


ui.brand()
page = st.sidebar.radio("menu", range(len(ui.PAGES)), key="page", label_visibility="collapsed",
                        format_func=lambda i: f"{ui.ICONS[i]} {ui.page_title(i)}")
if is_custom:
    st.sidebar.button(tr("العودة للبيانات التجريبية", "Back to the sample data"), on_click=use_sample_data,
                      key="back_side", icon=":material/undo:", width="stretch")
ui.sidebar_art()
ui.header(page, f"{ASOF:%Y-%m-%d}", is_custom)

PCT = ["breach_rate", "probability", "difference_vs_average", "previous_rate", "recent_rate", "change",
       "repeat_share", "facility_breach_180d", "actual_rate", "flag_rate", "recall", "precision",
       "precision_top20", "recall_top20", "next_rate_top8", "next_rate_all"]


def days_back(d):
    return ASOF - pd.Timedelta(days=d)


def period_select(options, default, key):
    st.session_state.setdefault(key, default)
    return st.selectbox(tr("الفترة", "Period"), options, key=key,
                        format_func=lambda d: tr(f"آخر {d} يوماً", f"Last {d} days"))


def with_drivers(frame):
    out = frame.copy()
    if "main_drivers" in out:
        out["main_drivers"] = out.main_drivers.map(lambda s: " + ".join(an.driver_names(s)))
    return out


def change(value, unit="", up_is_good=True):
    """Delta label and tone for a difference between two periods."""
    if pd.isna(value):
        return tr("لا فترة سابقة", "No earlier period"), "flat"
    if abs(value) < 1e-9:
        return tr("بلا تغيير", "No change"), "flat"
    arrow = "▲" if value > 0 else "▼"
    tone = "good" if (value > 0) == up_is_good else "bad"
    text = f"{abs(value):,.1f}" if isinstance(value, float) else f"{abs(value):,}"
    return f"{arrow} {text} {unit}".strip(), tone


def pct_text(value, digits=1):
    return "—" if pd.isna(value) else f"{value:.{digits}%}"


def no_model():
    ui.message(tr(*data["note"]), "warn")


def risk_name(code):
    ar, en, _ = ui.RISK[code]
    return tr(ar, en)


# ------------------------------------------------------------------ 0. about
def go_to(index):
    st.session_state.page = index


def about():
    ui.hero(tr("نموذج أولي لدعم القرار في إدارة المرافق", "A decision support prototype for facility management"),
            tr("قرارات استباقية لإدارة مرافق أذكى", "Proactive decisions for smarter facility management"),
            tr("يحوّل بيانات أوامر العمل إلى تنبؤ بخرق SLA، وترتيب للمخاطر، وتوصيات مدعومة بالأدلة يعتمدها مهندس التخطيط.",
               "It turns work order data into SLA breach predictions, risk rankings and evidence-based "
               "recommendations that a planning engineer approves."))
    ui.grid([ui.panel(tr("المشكلة", "The problem"), ui.listing([
                tr("التقارير التقليدية تصف ما حدث فقط", "Traditional reports only describe the past"),
                tr("التأخير يُكتشف بعد خرق SLA", "Delays are found after the SLA is breached"),
                tr("لا ترتيب واضح لأولويات التدخل", "No clear order of where to intervene")], True)),
             ui.panel(tr("الهدف", "The goal"), ui.listing([
                tr("إنذار مبكر عند إنشاء أمر العمل", "Early warning when a work order is created"),
                tr("توجيه المتابعة إلى المرافق والعقود الأخطر", "Focus follow-up on the riskiest sites and contracts"),
                tr("قرار أسرع بأرقام يمكن تتبعها", "Faster decisions with traceable numbers")])),
             ui.panel(tr("المستخدمون", "Who it is for"), ui.listing([
                tr("مهندس تخطيط المرافق", "Facility planning engineer"),
                tr("مدير العقد ومدير المرفق", "Contract and facility managers"),
                tr("الإدارة التنفيذية", "Executive management")]))], 3)
    ui.section(tr("ماذا يقدم التطبيق؟", "What does it offer?"), tr("اختر أي قسم لفتحه", "Open any section"))
    features = [
        (1, "📊", tr("المؤشرات والاتجاه وأداء المناطق في شاشة واحدة.", "KPIs, trend and regions on one screen.")),
        (2, "🎯", tr("احتمال تأخر كل أمر عمل مفتوح مع تنبيه.", "Late probability and alerts for open work orders.")),
        (3, "🛡️", tr("درجة مخاطر مفسّرة للمرافق والعقود.", "Explained risk scores for facilities and contracts.")),
        (4, "🤝", tr("مقارنة المقاولين مع تحفظاتها.", "Contractor comparison with its caveats.")),
        (5, "📝", tr("مذكرة قرار تُنزَّل بعد اعتماد بشري.", "A decision memo, downloadable after human approval.")),
        (6, "💬", tr("اسأل بالعربية أو الإنجليزية، والأرقام محسوبة من البيانات.", "Ask in Arabic or English; numbers come from data.")),
        (7, "📂", tr("ارفع أوامر جديدة، والبيانات الخاطئة تُرفض مع السبب.", "Upload new orders; bad rows are rejected with a reason.")),
        (8, "🗄️", tr("ارفع سجلك التاريخي فتتحدث كل الصفحات.", "Upload your history and every page updates.")),
        (9, "📘", tr("المنهجية ومعايير القبول وحدود النظام.", "Method, acceptance criteria and limits."))]
    for start in range(0, len(features), 3):
        for col, (index, icon, description) in zip(st.columns(3), features[start:start + 3]):
            with col:
                ui.html(ui.feature(icon, ui.page_title(index), description))
                st.button(tr("افتح", "Open"), key=f"go{index}", on_click=go_to, args=(index,), width="stretch")
    ui.section(tr("كيف يعمل؟", "How it works"))
    ui.steps([(tr("البيانات", "Data"), tr("أوامر العمل والمرافق ومدد SLA", "Work orders, facilities, SLA times")),
              (tr("فحص الجودة", "Quality check"), tr("تنظيف موثق دون تخمين", "Documented cleaning, no guessing")),
              (tr("الذكاء", "Intelligence"), tr("نموذج تنبؤ ودرجة مخاطر", "Prediction model and risk score")),
              (tr("التوصية", "Recommendation"), tr("إجراء ودليل ومالك", "Action, evidence, owner")),
              (tr("القرار", "Decision"), tr("الإنسان يعتمد", "A human approves"))])
    if is_custom:
        ui.message(tr("الصفحات تعرض الآن الملف الذي رفعته. دقة النتائج تعتمد على جودة بياناتك.",
                      "Pages now show the file you uploaded. Results are only as good as that data."), "warn")
    else:
        ui.message(tr("البيانات المستخدمة اصطناعية، والنتائج تثبت المنهجية ولا تمثل أي عقد أو جهة حقيقية.",
                      "All data is synthetic. Results demonstrate the method and represent no real contract or "
                      "organisation."), "warn")


# ------------------------------------------------------------------ 1. dashboard
def overview():
    c1, c2 = st.columns(2)
    with c1:
        days = period_select([30, 90, 180, 365], 90, "ov_days")
    region = c2.selectbox(tr("المنطقة", "Region"), ["all"] + sorted(data["fac"].region.unique()), key="ov_region",
                          format_func=lambda r: tr("كل المناطق", "All regions") if r == "all" else r)
    base = wo if region == "all" else wo[wo.region == region]
    now = an.kpis(an.period(base, days_back(days), ASOF))
    prev = an.kpis(an.period(base, days_back(2 * days), days_back(days)))
    risk = data["risk_f"] if region == "all" else data["risk_f"][data["risk_f"].region == region]
    vs = tr("عن السابقة", "vs previous")
    t1, _ = change(now["total"] - prev["total"], vs)
    t3, k3 = change(now["breaches"] - prev["breaches"], vs, up_is_good=False)
    t4, k4 = change(float((now["compliance"] - prev["compliance"]) * 100), tr("نقطة", "pts"))
    ui.grid([ui.kpi(tr("أوامر العمل", "Work orders"), f"{now['total']:,}", t1),
             ui.kpi(tr("الالتزام بـ SLA", "SLA compliance"), pct_text(now["compliance"]), t4, k4),
             ui.kpi(tr("خروقات SLA", "SLA breaches"), f"{now['breaches']:,}", t3, k3),
             ui.kpi(tr("أوامر مفتوحة", "Open work orders"), f"{now['open']:,}",
                    tr("المكتملة", "Completed") + f": {now['completed']:,}")])

    o = an.open_orders(data, model)
    if region != "all":
        o = o[o.region == region]
    regions = an.by_group(an.period(wo, days_back(days), ASOF), "region").dropna(subset=["breach_rate"])
    n_alert = int(o.alert.sum())
    cards = [] if model is None else [
        ui.insight(tr(f"{n_alert} أمر مفتوح يحتاج متابعة", f"{n_alert} open orders need follow-up"),
                   tr(f"فوق عتبة التنبيه من أصل {len(o)} أمر مفتوح", f"Above the alert threshold, of {len(o)} open"),
                   "bad" if n_alert else "good")]
    if len(risk):
        top_f = risk.iloc[0]
        cards.append(ui.insight(
            tr(f"أولى المرافق بالتدخل: {top_f.facility_id}", f"First facility to act on: {top_f.facility_id}"),
            tr(f"درجة مخاطر {top_f.risk_score:.0f} من 100 (آخر 90 يوماً)",
               f"Risk score {top_f.risk_score:.0f} of 100 (last 90 days)"), "warn"))
    if len(regions):
        cards.append(ui.insight(
        tr(f"أعلى منطقة خرقاً: {regions.index[0]}", f"Most breaches: {regions.index[0]}"),
        tr(f"{regions.breach_rate.iloc[0]:.0%} خلال آخر {days} يوماً",
           f"{regions.breach_rate.iloc[0]:.0%} over the last {days} days")))
    ui.grid(cards, 3)

    left, right = st.columns([2, 1])
    with left:
        ui.section(tr("اتجاه نسبة خرق SLA", "SLA breach rate trend"),
                   tr("شهرياً على كامل الفترة المتاحة", "Monthly, full history"))
        ui.trend(an.monthly(an.valid_rows(base)).breach_rate, dark)
    with right:
        ui.section(tr("توزيع الأولويات", "Priority mix"), tr(f"آخر {days} يوماً", f"Last {days} days"))
        ui.donut(an.period(base, days_back(days), ASOF).priority.value_counts().sort_index(), dark)
    ui.section(tr("أداء المناطق", "Regional performance"),
               tr("كل المناطق، مرتبة من الأعلى خرقاً", "All regions, highest breach rate first"))
    ui.bars(regions.breach_rate, dark)
    with st.expander(tr("عرض الجدول التفصيلي", "Show detailed table")):
        ui.table(regions, PCT)


# ------------------------------------------------------------------ 2. prediction
def prediction():
    if model is None:
        no_model()
        return
    t = data["test"]
    o = an.open_orders(data, model)
    tab1, tab2, tab3 = st.tabs([tr("التنبيهات الحالية", "Current alerts"), tr("أداء النموذج", "Model performance"),
                                tr("ما الذي يعتمد عليه", "What it relies on")])
    with tab1:
        ui.section(tr("الأوامر الأعلى احتمالاً للتأخر", "Most likely to be late"),
                   tr("الاحتمال تقديري وقت إنشاء الأمر، والقرار للمهندس",
                      "Estimated at creation time; the engineer decides"))
        ui.grid([ui.entity(r.wo_id, f"{r.priority} · {r.scope}", r.probability * 100,
                           "bad" if r.alert else "good", [r.facility_id, r.contractor_id],
                           [(tr("الاحتمال", "Probability"), f"{r.probability:.0%}"),
                            (tr("ساعات متبقية", "Hours left"), f"{r.hours_left:.0f}"),
                            (tr("تكرار الأصل", "Asset repeats"), int(r.asset_repeat_30d))])
                 for r in o.head(6).itertuples()], 3)
        ui.section(tr("كل الأوامر المفتوحة", "All open work orders"),
                   tr(f"{int(o.alert.sum())} فوق العتبة من {len(o)}", f"{int(o.alert.sum())} of {len(o)} above threshold"))
        ui.table(o[["wo_id", "probability", "alert", "priority", "scope", "facility_id", "contractor_id",
                    "hours_left", "contractor_backlog", "facility_breach_180d", "asset_repeat_30d"]],
                 PCT, index=False)
    with tab2:
        ui.grid([
            ui.kpi(tr("الخروقات المكتشفة (Recall)", "Breaches caught (recall)"), f"{t['recall']:.0%}",
                   tr(f"{t['TP']} من {t['test_breaches']}", f"{t['TP']} of {t['test_breaches']}"), "info"),
            ui.kpi(tr("دقة التنبيه (Precision)", "Alert precision"), f"{t['precision']:.0%}",
                   tr("إنذارات كاذبة", "False alarms") + f": {t['FP']}", "warn"),
            ui.kpi(tr("نسبة الأوامر المنبَّه عليها", "Share of orders flagged"), f"{t['flag_rate']:.0%}",
                   tr("فائتة", "Missed") + f": {t['FN']}"),
            ui.kpi(tr("عتبة التنبيه", "Alert threshold"), f"{t['threshold']:.3f}",
                   tr("اختيرت على فترة التحقق", "Chosen on validation"))])
        st.caption(tr(f"على فترة اختبار لم يرها النموذج: {t['test_rows']:,} أمر عمل.",
                      f"On a test period the model never saw: {t['test_rows']:,} work orders."))
        ui.section(tr("المقارنة بالقواعد البسيطة", "Against simple baselines"))
        ui.table(data["results"][["model", "PR_AUC", "ROC_AUC", "precision_top20", "recall_top20", "caught",
                                  "false_alarms", "missed"]], PCT, index=False)
        ui.section(tr("هل الاحتمالات واقعية؟", "Are the probabilities realistic?"),
                   tr("المتوقع مقابل الفعلي في خمس شرائح", "Predicted vs actual in five bands"))
        ui.grouped_bars(data["calibration"].set_index("band")[["predicted", "actual"]], dark)
    with tab3:
        ui.section(tr("أهم العوامل في النموذج", "Most important inputs"),
                   tr("Permutation importance على فترة التحقق", "Permutation importance on validation"))
        ui.bars(data["importance"].head(8).set_index("feature").importance, dark, fmt=".3f")
        ui.message(tr("أهمية العامل تعني أن النموذج يعتمد عليه، ولا تعني أنه سبب التأخر.",
                      "Importance means the model relies on an input, not that it causes delay."), "warn")


# ------------------------------------------------------------------ 3. risk
def risks():
    r = data["risk_f"]
    counts = r.risk_class.value_counts()
    ui.grid([ui.kpi(tr("خطورة مرتفعة", "High risk"), int(counts.get("High", 0)), tr("أولوية التدخل", "Act first"), "bad"),
             ui.kpi(tr("خطورة متوسطة", "Medium risk"), int(counts.get("Medium", 0)), tr("متابعة", "Watch"), "warn"),
             ui.kpi(tr("خطورة منخفضة", "Low risk"), int(counts.get("Low", 0)), tr("مراقبة", "Monitor"), "good"),
             ui.kpi(tr("مرافق مقيّمة", "Facilities scored"), len(r), tr("آخر 90 يوماً", "Last 90 days"))])
    tab1, tab2, tab3 = st.tabs([tr("المرافق", "Facilities"), tr("العقود", "Contracts"),
                                tr("منهجية الدرجة", "Score method")])
    with tab1:
        name = st.radio(tr("التصنيف", "Class"), ["High", "Medium", "Low"], format_func=risk_name,
                        horizontal=True, key="risk_class")
        view = r[r.risk_class == name]
        ui.grid([ui.entity(x.facility_id, f"{x.region} · {x.contract_id} · {x.facility_type}", x.risk_score,
                           ui.RISK[x.risk_class][2], an.driver_names(x.main_drivers),
                           [(tr("أوامر العمل", "Work orders"), int(x.work_orders)),
                            (tr("نسبة الخرق", "Breach rate"), f"{x.breach_rate:.0%}"),
                            (tr("التصنيف", "Class"), risk_name(x.risk_class))])
                 for x in view.head(9).itertuples()], 3)
        with st.expander(tr(f"كل المرافق في هذا التصنيف ({len(view)})", f"All facilities in this class ({len(view)})")):
            ui.table(with_drivers(view)[["facility_id", "risk_score", "risk_class", "main_drivers", "breach_rate",
                                         "work_orders", "region", "contract_id"]], PCT, index=False)
    with tab2:
        ui.grid([ui.entity(x.contract_id, tr("درجة مخاطر العقد (قاعدية)", "Rule-based contract risk"), x.risk_score,
                           ["bad", "warn", "good"][min(i, 2)], an.driver_names(x.main_drivers),
                           [(tr("أوامر العمل", "Work orders"), f"{int(x.work_orders):,}"),
                            (tr("نسبة الخرق", "Breach rate"), f"{x.breach_rate:.0%}"),
                            (tr("الترتيب", "Rank"), i + 1)]) for i, x in enumerate(data["risk_c"].itertuples())], 3)
        st.caption(tr("لون البطاقة يعبر عن الترتيب بين العقود، وليس عن حد مطلق.",
                      "Card colour shows the rank among contracts, not an absolute limit."))
    with tab3:
        ui.message(tr("درجة قاعدية (0-100) من خمسة مؤشرات بأوزان متساوية خلال 90 يوماً. ليست تنبؤ النموذج.",
                      "A rule-based score (0-100) from five equally weighted indicators over 90 days. "
                      "It is not the model's prediction."))
        ui.section(tr("التحقق: هل ترتّب الدرجة أداء الربع التالي؟", "Check: does the score rank next-quarter performance?"))
        v = data["risk_val"]
        if v.empty:
            ui.message(tr("التاريخ أو عدد المرافق لا يكفي للتحقق من الدرجة.",
                          "Not enough history or facilities to check the score."), "warn")
            return
        ui.table(v, PCT, index=False)
        mean = v.spearman_score_vs_next.mean()
        better = int((v.spearman_past_breach_only > v.spearman_score_vs_next).sum())
        ui.message(tr(f"متوسط ارتباط الدرجة بالفترة التالية {mean:.2f}، ونسبة الخرق السابقة وحدها تنبأت أفضل في "
                      f"{better} من {len(v)} فترات. استخدم الدرجة لترتيب الأولويات وفهم الأبعاد، لا كأداة تنبؤ.",
                      f"Mean rank correlation with the next period: {mean:.2f}. Past breach rate alone predicted better "
                      f"in {better} of {len(v)} periods. Use the score to prioritise and explain, not to predict."),
                   "warn")


# ------------------------------------------------------------------ 4. contractors
def contractors():
    days = period_select([90, 180, 365], 90, "ct_days")
    g = an.by_group(an.period(wo, days_back(days), ASOF), "contractor_id").dropna(subset=["breach_rate"])
    avg = g.breaches.sum() / g.known.sum()
    cards = []
    for cid, x in g.head(12).iterrows():
        tone = "bad" if x.breach_rate > avg * 1.2 else "good" if x.breach_rate < avg * 0.8 else "warn"
        label = {"bad": tr("أعلى من المتوسط", "Above average"), "warn": tr("قرب المتوسط", "Near average"),
                 "good": tr("أقل من المتوسط", "Below average")}[tone]
        cards.append(ui.entity(cid, tr("الالتزام بـ SLA", "SLA compliance"), (1 - x.breach_rate) * 100, tone, [],
                               [(tr("أوامر العمل", "Work orders"), f"{int(x.work_orders):,}"),
                                (tr("نسبة الخرق", "Breach rate"), f"{x.breach_rate:.1%}"),
                                (tr("مفتوحة", "Open"), int(x.open_now))], label))
    ui.grid(cards, 3)
    st.caption(tr(f"متوسط نسبة الخرق في الفترة: {avg:.1%}. التصنيف بفارق 20% عن المتوسط.",
                  f"Average breach rate in the period: {avg:.1%}. Labels use a 20% band around it."))
    tab1, tab2, tab3 = st.tabs([tr("الاتجاه", "Trend"), tr("التفاصيل", "Details"),
                                tr("عدالة التنبيهات", "Alert fairness")])
    with tab1:
        ui.lines(an.monthly(an.valid_rows(wo), "contractor_id"), dark)
    with tab2:
        ui.table(g, PCT)
    with tab3:
        if model is None:
            no_model()
        else:
            ui.table(data["groups"], PCT, index=False)
            missed = data["groups"][data["groups"].recall < 0.2].contractor_id.tolist()
            if missed:
                names = "، ".join(map(str, missed[:6]))
                ui.message(tr(f"النموذج يغفل معظم خروقات: {names}. يلزم متابعتهم بطريقة أخرى.",
                              f"The model misses most breaches of: {', '.join(map(str, missed[:6]))}. "
                              "They need another follow-up method."), "error")
    ui.message(tr("المقارنة ارتباطية: تختلف المرافق ونوع الأعمال وحجمها بين المقاولين.",
                  "This is a descriptive comparison: facilities, work mix and volume differ between contractors."), "warn")


# ------------------------------------------------------------------ 5. decision memo
def memo():
    days = period_select([30, 90, 180], 90, "memo_days")
    now = an.kpis(an.period(wo, days_back(days), ASOF))
    prev = an.kpis(an.period(wo, days_back(2 * days), days_back(days)))
    recs = an.recommendations(data, model, days)
    title = tr(f"مذكرة قرار: أداء أوامر العمل خلال آخر {days} يوماً",
               f"Decision memo: work order performance, last {days} days")
    kind = tr("ملف مرفوع من المستخدم", "user-uploaded file") if is_custom else tr("بيانات اصطناعية", "synthetic data")
    meta = tr(f"تاريخ البيانات: {ASOF:%Y-%m-%d} · {kind} · المقارنة مع الـ {days} يوماً السابقة",
              f"Data as of {ASOF:%Y-%m-%d} · {kind} · compared with the previous {days} days")
    ui.memo_head(title, meta)
    a, _ = change(now["total"] - prev["total"], tr("أمر", "orders"))
    b, kb = change(now["breaches"] - prev["breaches"], tr("خرق", "breaches"), up_is_good=False)
    c, kc = change(float((now["compliance"] - prev["compliance"]) * 100), tr("نقطة", "pts"))
    compare = [ui.compare_card(tr("أوامر العمل", "Work orders"), f"{now['total']:,}", f"{prev['total']:,}", a, "flat"),
               ui.compare_card(tr("خروقات SLA", "SLA breaches"), f"{now['breaches']:,}", f"{prev['breaches']:,}", b, kb),
               ui.compare_card(tr("الالتزام بـ SLA", "SLA compliance"), pct_text(now["compliance"]),
                               pct_text(prev["compliance"]), c, kc)]
    ui.grid(compare, 3)
    st.caption(tr(f"الرقم المشطوب هو الفترة السابقة. الأوامر المفتوحة ضمن مدة SLA ({now['open']}) لا تدخل في الالتزام.",
                  f"Struck-through figures are the previous period. Open orders still within SLA ({now['open']}) "
                  "are not counted in compliance."))

    ui.section(tr("الإجراءات المقترحة", "Proposed actions"),
               tr(f"{len(recs)} إجراءات مرتبة حسب الأولوية، كل إجراء بدليله ومالكه",
                  f"{len(recs)} actions by priority, each with evidence and owner"))
    if recs.empty:
        recs = pd.DataFrame(columns=["priority", "action", "evidence", "owner", "ids"])
    options = ["all"] + [p for p in ui.PRIORITY if p in set(recs.priority)]
    pick = st.radio(tr("عرض الأولوية", "Show priority"), options, horizontal=True, key="memo_priority",
                    format_func=lambda p: tr("الكل", "All") if p == "all" else tr(*ui.PRIORITY[p][:2]))
    ui.actions(recs if pick == "all" else recs[recs.priority == pick])

    t = data["test"]
    model_note = (tr("لا يوجد نموذج تنبؤ لهذه البيانات.", "There is no prediction model for this data.") if model is None
                  else tr(f"دقة تنبيه النموذج في الاختبار {t['precision']:.0%} ويكتشف {t['recall']:.0%} من الخروقات.",
                          f"In testing, alert precision was {t['precision']:.0%} and the model caught "
                          f"{t['recall']:.0%} of breaches."))
    data_note = (tr("النتائج من ملف رفعه المستخدم ولم تُراجع جودته خارج الفحص الآلي.",
                    "Results come from a user file checked only by automatic rules.") if is_custom
                 else tr("البيانات اصطناعية ولا تمثل عقداً حقيقياً.", "The data is synthetic and represents no real contract."))
    caveats = [tr("العلاقات ارتباطية ولا تثبت سبباً.", "Relationships are associations, not proven causes."), model_note,
               tr("لا يُنفذ أي إجراء دون موافقة بشرية.", "No action is taken without human approval."), data_note]
    left, right = st.columns(2)
    with left:
        ui.html(ui.panel(tr("تحفظات", "Caveats"), ui.listing(caveats, True)))
    with right:
        with st.container(border=True):
            ui.section(tr("الاعتماد البشري", "Human approval"),
                       tr("التنزيل يتطلب اسم المراجع وموافقته", "Download needs a reviewer name and consent"))
            name = st.text_input(tr("اسم المراجع", "Reviewer name"), key="reviewer",
                                 placeholder=tr("اكتب اسمك", "Type your name"))
            ok = st.checkbox(tr("راجعتُ الأرقام والأدلة وأعتمد المسودة",
                                "I reviewed the figures and evidence and approve the draft"), key="approve")
            if ok and name.strip():
                doc = ui.memo_document(title, meta, compare, recs, caveats, name.strip())
                st.download_button(tr("تنزيل المذكرة المعتمدة (للطباعة)", "Download approved memo (printable)"), doc,
                                   file_name=f"decision_memo_{days}d_{code}.html", mime="text/html",
                                   icon=":material/download:")
            else:
                ui.message(tr("بانتظار الاعتماد.", "Waiting for approval."))


# ------------------------------------------------------------------ 6. assistant
def assistant():
    history = st.session_state.setdefault("chat", [])
    st.caption(tr("مساعد قائم على قواعد: يفهم الأسئلة بالكلمات المفتاحية، وكل رقم محسوب من البيانات. لا يستخدم LLM.",
                  "Rule-based assistant: it matches keywords and computes every number from the data. No LLM."))
    asked = None
    ui.section(tr("أسئلة مقترحة", "Suggested questions"))
    cols = st.columns(3)
    for i, q in enumerate(an.suggestions()):
        if cols[i % 3].button(q, key=f"s{i}", width="stretch"):
            asked = q
    with st.form("ask", clear_on_submit=True):
        typed = st.text_input(tr("سؤالك", "Your question"), key="question",
                              placeholder=tr("مثال: ما المقاولون الأعلى تأخيراً خلال 180 يوم؟",
                                             "Example: which contractors were latest over 180 days?"))
        if st.form_submit_button(tr("اسأل", "Ask"), icon=":material/send:") and typed.strip():
            asked = typed.strip()
    if asked:
        history.insert(0, asked)
        del history[5:]
    for n, q in enumerate(history):  # newest first
        answer, table, source = an.copilot(q, data, model)
        ui.bubble(q, "me")
        ui.bubble(answer, "bot")
        if n == 0 and table is not None and len(table):
            if "action" in table:
                ui.actions(table)
            else:
                ui.table(with_drivers(table), PCT, index=table.index.name is not None)
        st.caption(tr("المصدر", "Source") + f": {source}")
    if history and st.button(tr("مسح المحادثة", "Clear chat"), key="clear", icon=":material/delete:"):
        history.clear()
        st.rerun()


# ------------------------------------------------------------------ 7. upload check
def upload():
    if model is None:
        no_model()
        return
    ui.steps([(tr("جهّز الملف", "Prepare"), tr("CSV بالأعمدة الإلزامية", "CSV with required columns")),
              (tr("ارفعه", "Upload"), tr("أو شغّل مثالاً جاهزاً", "Or run a ready example")),
              (tr("فحص تلقائي", "Validate"), tr("الصف الخاطئ يُرفض مع السبب", "Bad rows are rejected with a reason")),
              (tr("النتيجة", "Result"), tr("احتمال التأخر لكل أمر", "Late probability per order"))])
    st.markdown(tr("الأعمدة الإلزامية: ", "Required columns: ")
                + "`wo_id, created_at, priority, scope, facility_id` "
                + tr("والاختياري", "and optional") + " `asset_id`.")
    if is_custom:
        ui.message(tr("استخدم أرقام المرافق والنطاقات الموجودة في سجلك، وتواريخ قريبة من آخر تاريخ فيه.",
                      "Use facility IDs and scopes from your history, dated near its latest date."))
    c = st.columns(2)
    samples = [] if is_custom else [(tr("ملف تجريبي سليم", "Valid sample file"), "samples/new_orders_ok.csv"),
                                    (tr("ملف تجريبي به أخطاء", "Sample file with errors"),
                                     "samples/new_orders_errors.csv")]
    for col, (label, path) in zip(c, samples):
        with open(path, "rb") as f:
            col.download_button(label, f.read(), file_name=os.path.basename(path), icon=":material/download:",
                                width="stretch")
    names = {"none": tr("لا", "None"), "ok": tr("السليم", "Valid"), "err": tr("الذي به أخطاء", "With errors")}
    demo = "none" if is_custom else st.radio(tr("أو شغّل مثالاً مباشرة", "Or run an example"), list(names),
                                             format_func=names.get, horizontal=True, key="demo")
    file = st.file_uploader(tr("ملف CSV", "CSV file"), type="csv", key="upload_file")
    source = file
    if source is None and demo != "none":
        source = "samples/new_orders_ok.csv" if demo == "ok" else "samples/new_orders_errors.csv"
    if source is None:
        return
    try:
        up = pd.read_csv(source, dtype=str, encoding="utf-8-sig")
    except Exception:
        ui.message(tr("تعذرت قراءة الملف. تأكد أنه CSV بترميز UTF-8.", "Could not read the file. Use a UTF-8 CSV."), "error")
        return
    if len(up) > 2000:
        ui.message(tr("الحد الأقصى 2000 صف في المرة الواحدة.", "Maximum 2000 rows at a time."), "error")
        return
    good, bad, fatal = an.validate_upload(up, data)
    if fatal:
        ui.message(fatal, "error")
        return
    res = an.score_new(good, data, model) if len(good) else None
    ui.grid([ui.kpi(tr("صفوف مقروءة", "Rows read"), len(up)),
             ui.kpi(tr("مقبولة", "Accepted"), len(good), tr("دخلت التنبؤ", "Scored"), "good"),
             ui.kpi(tr("مرفوضة", "Rejected"), len(bad), tr("تحتاج تصحيحاً", "Need fixing"),
                    "bad" if len(bad) else "flat"),
             ui.kpi(tr("فوق عتبة التنبيه", "Above threshold"), int(res.alert.sum()) if res is not None else 0,
                    tr("من المقبولة", "Of accepted"), "warn")])
    if len(bad):
        ui.section(tr("صفوف مرفوضة", "Rejected rows"),
                   tr("صحّحها وأعد الرفع. لا تُخمَّن أي قيمة", "Fix and upload again. No value is guessed"))
        ui.table(bad[["wo_id", "rejection_reason"]], index=False)
    if res is not None:
        ui.section(tr("نتائج التنبؤ", "Predictions"))
        ui.table(res[["wo_id", "probability", "alert", "priority", "facility_id", "contractor_id",
                      "contractor_backlog", "facility_breach_180d", "asset_repeat_30d"]], PCT, index=False)
        st.caption(tr("الأعمدة الأخيرة سياق للمراجعة وليست تفسيراً سببياً للاحتمال.",
                      "The last columns are context for review, not a causal explanation."))


# ------------------------------------------------------------------ 8. my data
def apply_history(good):
    """Analyse the history and make it the active data of this session. Returns an error message or None."""
    try:
        new_data, new_model = cd.build_dataset(good, boot()[0]["sla"])
    except Exception as error:  # show the reason instead of a crash page
        return str(error)
    reset_choices()
    st.session_state.custom = {"data": new_data, "model": new_model}
    return None


def my_data():
    ui.steps([(tr("جهّز السجل", "Prepare"), tr("CSV لأوامر عمل سابقة", "CSV of past work orders")),
              (tr("ارفعه", "Upload"), tr("أو جرّب السجل التجريبي", "Or try the sample history")),
              (tr("فحص تلقائي", "Validate"), tr("الصف الخاطئ يُرفض مع السبب", "Bad rows are rejected with a reason")),
              (tr("طبّق", "Apply"), tr("كل الصفحات تتحدث لجلستك", "Every page updates for your session"))])
    if is_custom:
        ui.message(tr(f"كل الصفحات تعرض الآن سجلك ({len(wo):,} أمر عمل، حتى ⁦{ASOF:%Y-%m-%d}⁩).",
                      f"Every page now shows your history ({len(wo):,} work orders, up to {ASOF:%Y-%m-%d})."), "ok")
        if model is None:
            no_model()
        st.button(tr("العودة للبيانات التجريبية", "Back to the sample data"), on_click=use_sample_data,
                  key="back_page", icon=":material/undo:")
    ui.message(tr("تنبيه: هذا رابط عام على خادم خارجي. لا ترفع بيانات حقيقية أو سرية من جهة عملك. استخدم بيانات "
                  "اصطناعية، أو مجهولة الهوية وبتصريح. الملف لا يُحفظ ويزول بإغلاق الصفحة أو تحديثها.",
                  "Warning: this is a public link on an external server. Do not upload real or confidential company "
                  "data; use synthetic data, or anonymised data you are authorised to share. The file is not saved "
                  "and disappears when the page is closed or reloaded."), "error")
    st.markdown(tr("الأعمدة الإلزامية: ", "Required columns: ") + "`" + ", ".join(cd.REQUIRED) + "`  \n"
                + tr("الاختيارية: ", "Optional: ") + "`" + ", ".join(cd.OPTIONAL) + "`  \n"
                + tr("التاريخ بصيغة YYYY-MM-DD HH:MM، والأولوية P1 إلى P4، و`completed_at` فارغ للأمر المفتوح.",
                     "Dates as YYYY-MM-DD HH:MM, priority P1 to P4, `completed_at` empty for open orders."))
    c = st.columns(2)
    for col, (label, path) in zip(c, [(tr("قالب الأعمدة", "Column template"), "samples/history_template.csv"),
                                      (tr("سجل تجريبي كامل", "Full sample history"), "samples/history_sample.csv")]):
        with open(path, "rb") as f:
            col.download_button(label, f.read(), file_name=os.path.basename(path), icon=":material/download:",
                                width="stretch")
    kinds = {"file": tr("ملفي", "My file"), "sample": tr("السجل التجريبي (اصطناعي)", "Synthetic sample history")}
    sample = st.radio(tr("مصدر السجل", "History source"), list(kinds), format_func=kinds.get, horizontal=True,
                      key="history_source") == "sample"
    file = None if sample else st.file_uploader(tr("سجل أوامر العمل (CSV)", "Work order history (CSV)"), type="csv",
                                                key="history_file")
    source = "samples/history_sample.csv" if sample else file
    if source is None:
        return
    try:
        raw = pd.read_csv(source, dtype=str, encoding="utf-8-sig")
    except Exception:
        ui.message(tr("تعذرت قراءة الملف. تأكد أنه CSV بترميز UTF-8.", "Could not read the file. Use a UTF-8 CSV."), "error")
        return
    good, bad, fatal = cd.validate_history(raw, boot()[0]["sla"])
    if good is not None:
        ui.grid([ui.kpi(tr("صفوف مقروءة", "Rows read"), f"{len(raw):,}"),
                 ui.kpi(tr("مقبولة", "Accepted"), f"{len(good):,}", "", "good"),
                 ui.kpi(tr("مرفوضة", "Rejected"), f"{len(bad):,}",
                        tr("مع سبب كل صف", "With a reason each") if len(bad) else "", "bad" if len(bad) else "flat")])
        if len(bad):
            ui.table(bad.head(100), index=False)
    if fatal:
        ui.message(fatal, "error")
        return
    consent = sample or st.checkbox(tr("أؤكد أن الملف لا يحوي بيانات سرية، أو أني مخوّل برفعه بعد إخفاء الهوية",
                                       "I confirm the file holds no confidential data, or I am authorised to share "
                                       "this anonymised copy"), key="consent")
    if st.button(tr("طبّق على كل الصفحات", "Apply to every page"), key="apply", disabled=not consent,
                 icon=":material/check_circle:", type="primary"):
        with st.spinner(tr("جارٍ التحليل وتدريب النموذج...", "Analysing the history and training the model...")):
            error = apply_history(good)
        if error:
            ui.message(error, "error")
        else:
            st.rerun()


# ------------------------------------------------------------------ 9. method
def method():
    t = data["test"]
    count = (tr(f"{len(wo):,} أمر عمل من ملفك", f"{len(wo):,} work orders from your file") if is_custom
             else tr("24 ألف أمر عمل اصطناعي", "24k synthetic work orders"))
    ui.steps([(tr("البيانات", "Data"), count),
              (tr("الجودة", "Quality"), tr("فحص وتنظيف موثق", "Documented cleaning")),
              (tr("الخصائص", "Features"), tr("المتاح وقت الإنشاء فقط", "Known at creation only")),
              (tr("النموذج", "Model"), tr("تنبؤ + درجة مخاطر", "Prediction + risk score")),
              (tr("التوصيات", "Actions"), tr("إجراء ودليل ومالك", "Action, evidence, owner")),
              (tr("مراجعة بشرية", "Human review"), tr("لا تنفيذ دون موافقة", "Nothing without approval"))])
    panels = []
    if model is None:
        no_model()
    else:
        panels.append(method_model(t))
    panels += [
        ui.panel(tr("معايير القبول", "Acceptance criteria"), ui.listing([
            tr("كل رقم محسوب من ملفات البيانات", "Every number is computed from data files"),
            tr("البيانات المفقودة تُرفض ولا تُخمَّن", "Missing data is rejected, never guessed"),
            tr("السؤال خارج البيانات يُرفض", "Out-of-data questions are refused"),
            tr("لا اعتماد دون مراجعة بشرية", "No approval without human review"),
            tr("مصدر البيانات ظاهر في كل صفحة", "The data source is shown on every page")])),
        ui.panel(tr("الحدود", "Limits"), ui.listing([
            tr("نتائج الملف المرفوع تعتمد على جودته", "Results depend on the quality of the uploaded file")
            if is_custom else tr("العلاقات في البيانات مزروعة؛ النتائج تثبت المنهجية فقط",
                                 "Relationships are planted; results prove the method only"),
            tr("قد تتفاوت التنبيهات بين المقاولين", "Alerts may be uneven across contractors"),
            tr("الارتباط لا يثبت السببية", "Correlation does not prove cause"),
            tr("أوامر العمل Corrective فقط", "Corrective work orders only"),
            tr("المساعد يفهم أنواعاً محددة من الأسئلة", "The assistant handles specific question types")], True))]
    ui.grid(panels, 3)


def method_model(t):
    """Show the test figures and return the model panel."""
    pr = data["results"].PR_AUC
    p = data["periods"] or {"train": tr("أكتوبر 2024 - ديسمبر 2025", "Oct 2024 - Dec 2025"),
                            "validation": tr("يناير - مارس 2026", "Jan - Mar 2026"),
                            "test": tr("أبريل - يونيو 2026", "Apr - Jun 2026")}
    ui.grid([ui.kpi(tr("PR-AUC في الاختبار", "Test PR-AUC"), f"{pr.iloc[-1]:.3f}",
                    tr("القاعدة البسيطة", "Simple rule") + f": {pr.iloc[1]:.3f}", "info"),
             ui.kpi(tr("الاكتشاف", "Recall"), f"{t['recall']:.0%}", tr("من الخروقات", "Of breaches")),
             ui.kpi(tr("دقة التنبيه", "Alert precision"), f"{t['precision']:.0%}",
                    tr("من التنبيهات الصحيحة", "Of alerts that were right"), "warn"),
             ui.kpi(tr("اختبار التسرب", "Leakage test"), "—" if is_custom else "PASS",
                    tr("نفس الكود، دون اختبار على ملفك", "Same code, not re-tested on your file") if is_custom
                    else tr("لا معلومات مستقبلية", "No future data"), "flat" if is_custom else "good")])
    return ui.panel(tr("النموذج والتحقق", "Model and validation"), ui.listing([
        t["chosen_model"] + ("" if is_custom else " (Gradient Boosting)"),
        tr("التدريب: ", "Training: ") + p["train"],
        tr("الاختيار على التحقق: ", "Selection on validation: ") + p["validation"],
        tr("الاختبار على فترة لاحقة: ", "Tested on a later period: ") + p["test"]]))


[about, overview, prediction, risks, contractors, memo, assistant, upload, my_data, method][page]()
