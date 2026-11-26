"""
ui.py - design system: light/dark tokens, right-to-left / left-to-right CSS, components and charts.
All text coming from data or uploads is HTML-escaped.
"""
import base64
import io
import os
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from i18n import lang, tr

ICONS = [":material/home:", ":material/space_dashboard:", ":material/notifications_active:", ":material/shield:",
         ":material/groups:", ":material/assignment:", ":material/forum:", ":material/upload_file:",
         ":material/menu_book:"]
PAGES = [  # (arabic title, english title, arabic subtitle, english subtitle)
    ("نظرة عامة", "Overview", "ما هذا التطبيق وماذا يقدم؟", "What this application is and what it offers"),
    ("لوحة الأداء", "Performance dashboard", "أين نقف الآن، وكيف تغيّر الأداء؟", "Where we stand and how performance moved"),
    ("التنبؤ والتنبيهات", "Predictions and alerts", "أي أوامر العمل تحتاج متابعة قبل أن تتأخر؟", "Which work orders need action before they are late"),
    ("مخاطر المرافق والعقود", "Facility and contract risk", "أي المرافق والعقود أولى بالتدخل؟", "Where to intervene first"),
    ("أداء المقاولين", "Contractor performance", "مقارنة المقاولين مع تحفظاتها", "Contractor comparison, with its caveats"),
    ("مذكرة القرار والتوصيات", "Decision memo", "من الأرقام إلى قرار يعتمده إنسان", "From numbers to a human-approved decision"),
    ("المساعد", "Assistant", "اسأل عن بياناتك بالعربية أو الإنجليزية", "Ask about your data in Arabic or English"),
    ("فحص ملف جديد", "Check a new file", "تنبؤ لأوامر جديدة مع رفض البيانات الخاطئة", "Score new work orders and reject bad data"),
    ("المنهجية والحدود", "Method and limits", "كيف بُني النظام وأين حدوده", "How it was built and where it stops"),
]

THEMES = {
    False: dict(bg="#f4f7f6", panel="#ffffff", panel2="#edf3f1", text="#0f1f24", muted="#5b6f75",
                border="#dfe9e6", accent="#0f766e", accent2="#0ea5e9", soft="#e2f4f0", grid="#e6eeec",
                good="#047857", goodbg="#e3f7ee", warn="#b45309", warnbg="#fdf1dc", bad="#be123c",
                badbg="#fde8ee", shadow="0 1px 2px rgba(15,31,36,.05),0 10px 28px -14px rgba(15,31,36,.16)",
                glow="rgba(15,118,110,.12)"),
    True: dict(bg="#081113", panel="#0f1b1f", panel2="#16262b", text="#e8f3f1", muted="#8aa3a8",
               border="#1f343a", accent="#2dd4bf", accent2="#38bdf8", soft="#103330", grid="#1b2d32",
               good="#4ade80", goodbg="#0e2d20", warn="#fbbf24", warnbg="#322710", bad="#fb7185",
               badbg="#36131d", shadow="0 1px 2px rgba(0,0,0,.4),0 14px 34px -18px rgba(0,0,0,.75)",
               glow="rgba(45,212,191,.13)"),
}
SERIES = ["#14b8a6", "#f59e0b", "#6366f1", "#f43f5e", "#0ea5e9", "#94a3b8"]
PRIORITY = {"urgent": ("عاجل", "Urgent", "bad"), "high": ("مرتفع", "High", "warn"),
            "medium": ("متوسط", "Medium", "info")}
RISK = {"High": ("مرتفع", "High", "bad"), "Medium": ("متوسط", "Medium", "warn"), "Low": ("منخفض", "Low", "good")}

LABELS = {
    "wo_id": ("أمر العمل", "Work order"), "priority": ("الأولوية", "Priority"), "scope": ("النطاق", "Scope"),
    "facility_id": ("المرفق", "Facility"), "contractor_id": ("المقاول", "Contractor"),
    "contract_id": ("العقد", "Contract"), "region": ("المنطقة", "Region"),
    "probability": ("احتمال التأخر", "Late probability"), "alert": ("تنبيه", "Alert"),
    "hours_left": ("ساعات متبقية", "Hours left"), "contractor_backlog": ("تراكم المقاول", "Contractor backlog"),
    "facility_breach_180d": ("خرق المرفق (180 يوم)", "Facility breach (180d)"),
    "asset_repeat_30d": ("تكرار الأصل (30 يوم)", "Asset repeats (30d)"),
    "work_orders": ("أوامر العمل", "Work orders"), "known": ("معلومة النتيجة", "Known outcome"),
    "breaches": ("الخروقات", "Breaches"), "breach_rate": ("نسبة الخرق", "Breach rate"),
    "risk_score": ("درجة المخاطر", "Risk score"), "risk_class": ("التصنيف", "Class"),
    "main_drivers": ("أبرز العوامل", "Main drivers"), "median_time_vs_sla": ("وسيط الزمن / SLA", "Median time / SLA"),
    "median_response_h": ("وسيط الاستجابة (ساعة)", "Median response (h)"),
    "repeat_share": ("نسبة التكرار", "Repeat share"), "open_now": ("مفتوحة", "Open"),
    "model": ("الطريقة", "Method"), "PR_AUC": ("PR-AUC", "PR-AUC"), "ROC_AUC": ("ROC-AUC", "ROC-AUC"),
    "precision_top20": ("الدقة (أعلى 20%)", "Precision (top 20%)"),
    "recall_top20": ("الاكتشاف (أعلى 20%)", "Recall (top 20%)"), "caught": ("مكتشفة", "Caught"),
    "false_alarms": ("إنذارات كاذبة", "False alarms"), "missed": ("فائتة", "Missed"),
    "as_of": ("تاريخ القياس", "As of"),
    "spearman_score_vs_next": ("ارتباط الدرجة بالربع التالي", "Score vs next quarter (rank corr.)"),
    "spearman_past_breach_only": ("ارتباط نسبة الخرق وحدها", "Breach rate alone (rank corr.)"),
    "next_rate_top8": ("خرق أعلى 8 لاحقاً", "Next breach rate, top 8"),
    "next_rate_all": ("خرق الجميع لاحقاً", "Next breach rate, all"), "factor": ("العامل", "Factor"),
    "difference_vs_average": ("الفرق عن المتوسط", "Vs average"), "previous_rate": ("الخرق سابقاً", "Previous rate"),
    "recent_rate": ("الخرق حالياً", "Recent rate"), "previous_n": ("العدد سابقاً", "Previous n"),
    "recent_n": ("العدد حالياً", "Recent n"), "change": ("التغير", "Change"),
    "beyond_noise": ("يتجاوز التذبذب", "Beyond noise"), "asset_id": ("الأصل", "Asset"),
    "reports": ("البلاغات", "Reports"), "n": ("العدد", "Count"), "actual_rate": ("الخرق الفعلي", "Actual rate"),
    "flag_rate": ("نسبة التنبيه", "Flag rate"), "recall": ("الاكتشاف", "Recall"),
    "precision": ("الدقة", "Precision"), "rejection_reason": ("سبب الرفض", "Rejection reason"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Alexandria:wght@500;600;700;800&family=Tajawal:wght@400;500;700;800&display=swap');
.stApp {background: radial-gradient(900px 420px at 85% -120px, @glow, transparent 70%), @bg; color: @text;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp div, .stApp td, .stApp th, .stApp a, [role="option"] {font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif;}
.stApp [data-testid="stIconMaterial"], .stApp span[translate="no"] {font-family: 'Material Symbols Rounded' !important;}
[data-testid="stMain"], [data-testid="stSidebarContent"] {direction: @dir; text-align: start;}
[data-testid="stSidebarHeader"] {direction: ltr;}
.stApp [data-testid="stMarkdownContainer"] {direction: @dir; text-align: @align; width: 100%;}
.stApp .section, .stApp .hero .h, .stApp .hero .p {text-align: @align; width: 100%;}
.stApp .stButton [data-testid="stMarkdownContainer"] {text-align: center; width: auto;}
header[data-testid="stHeader"] {background: transparent; pointer-events: none;} /* the empty bar must not swallow clicks */
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, [data-testid="stExpandSidebarButton"] {pointer-events: auto;}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg {color: @text !important;}
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] * {color: @text !important; opacity: 1 !important; visibility: visible !important;}
.block-container {padding-top: 4.25rem; max-width: 1220px;} /* below the 60px header */
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {color: @text;}
[data-testid="stCaptionContainer"] p {color: @muted;}
.stApp code {background: @soft; color: @accent; border-radius: 6px;}
section[data-testid="stSidebar"] {background: @panel; border-right: 1px solid @border;}
section[data-testid="stSidebar"] [data-testid="stElementContainer"], section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div {width: 100% !important;}
section[data-testid="stSidebar"] [role="radiogroup"] {gap: .2rem; width: 100%; align-items: stretch;}
section[data-testid="stSidebar"] [role="radiogroup"] label {width: 100%; margin: 0; padding: .62rem .8rem; border-radius: 12px; transition: background .15s;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]) {display: none;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: @panel2;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: @soft; box-shadow: inset @edge 0 0 @accent;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color: @accent; font-weight: 700;}
section[data-testid="stSidebar"] [role="radiogroup"] p {font-size: .93rem; display: flex; gap: .6rem; align-items: center;}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"], .stTextInput input, .stTextInput div[data-testid="stTextInputRootElement"], [data-testid="stSelectbox"] div[role="group"], [data-testid="stFileUploaderDropzone"] {background: @panel !important; color: @text !important; border-color: @border !important; border-radius: 12px !important;}
[data-testid="stSelectbox"] input, div[data-baseweb="select"] span {color: @text !important;}
[data-testid="stSelectbox"] svg {fill: @muted;}
.stApp input::placeholder {color: @muted !important; -webkit-text-fill-color: @muted !important; opacity: 1;}
[data-testid="stFileUploaderDropzone"] {border: 1.5px dashed @accent !important; padding: 1.1rem;}
[data-testid="stFileUploaderDropzone"] * {color: @muted !important;}
div[data-baseweb="popover"] *, [role="listbox"], [role="option"] {background-color: @panel; color: @text;}
[role="listbox"], [role="option"] {direction: @dir; text-align: start;}
[role="option"]:hover, [role="option"]:hover * {background-color: @soft;}
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stFileUploaderDropzone"] button {background: @panel !important; color: @text !important; border: 1px solid @border !important; border-radius: 12px; transition: all .15s; box-shadow: @shadow;}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {border-color: @accent !important; transform: translateY(-1px);}
.stButton button:hover p, .stDownloadButton button:hover p {color: @accent;}
[data-testid="stForm"] {border: 1px solid @border; border-radius: 18px; background: @panel; box-shadow: @shadow;}
.stCheckbox label > div:first-of-type {direction: ltr;}
[role="tablist"] {gap: .35rem; background: @panel2; padding: .3rem; border-radius: 14px; border: 1px solid @border; width: fit-content; max-width: 100%; overflow-x: auto;}
[role="tab"] {border-radius: 10px !important; padding: .35rem 1rem !important; height: auto !important; border: 0 !important;}
[role="tab"] p {color: @muted; font-weight: 600; white-space: nowrap;}
[role="tab"][aria-selected="true"] {background: @panel !important; box-shadow: @shadow;}
[role="tab"][aria-selected="true"] p {color: @accent;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator {display: none !important;}
[data-testid="stExpander"] details {border: 1px solid @border; border-radius: 14px; background: @panel;}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .vega-embed {direction: ltr;}
.head {display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; margin: .2rem 0 1.2rem;}
.head .t {font-size: 2rem; font-weight: 700; line-height: 1.35;}
.head .s {color: @muted; font-size: .98rem;}
.stApp .asof {color: @muted; font-size: .74rem;}
/* theme switch sits at the far end of the top row, opposite the language choice */
.st-key-theme, .st-key-theme > div {width: 100% !important; display: flex; flex-direction: column; align-items: flex-end;}
.st-key-theme [role="radiogroup"] {justify-content: flex-end;}
.brand {display: flex; gap: .7rem; align-items: center; margin-bottom: 1rem;}
.stApp .brand .logo {width: 52px; height: 52px; flex: none; border-radius: 14px; box-shadow: 0 8px 20px -8px #0f766e;}
.brand b {display: block; font-size: .95rem; color: @text;}
.brand small {color: @muted; font-size: .72rem;}
.art {margin: .4rem 0 0; text-align: center;}
.art img {width: 100%; max-width: 300px; height: auto; border-radius: 18px; box-shadow: @shadow; border: 1px solid @border;}
.art figcaption {color: @muted; font-size: .86rem; margin-top: .45rem; text-align: center;}.grid {display: grid; gap: .9rem; margin: .2rem 0 1.1rem;}
.g4 {grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));}
.g3 {grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));}
.card {background: @panel; border: 1px solid @border; border-radius: 18px; padding: 1.05rem 1.15rem; box-shadow: @shadow; min-width: 0;}
.kpi .l {color: @muted; font-size: .84rem; font-weight: 500;}
.kpi .v {font-size: 2.05rem; font-weight: 700; direction: ltr; text-align: @align; line-height: 1.45; font-variant-numeric: tabular-nums;}
.delta {display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .76rem; font-weight: 600;}
.good {color: @good; background: @goodbg;} .bad {color: @bad; background: @badbg;}
.warn {color: @warn; background: @warnbg;} .info {color: @accent; background: @soft;} .flat {color: @muted; background: @panel2;}
.insight {display: flex; gap: .8rem; align-items: flex-start;}
.dot {width: 10px; height: 10px; border-radius: 50%; margin-top: .55rem; flex: none;}
.dot.bad {background: @bad;} .dot.warn {background: @warn;} .dot.good {background: @good;} .dot.info {background: @accent;}
.insight b {display: block; font-size: .95rem;}
.insight span {color: @muted; font-size: .84rem;}
.section {font-size: 1.08rem; font-weight: 700; margin: 1.3rem 0 .6rem;}
.section small {display: block; color: @muted; font-weight: 400; font-size: .82rem;}
.entity .top {display: flex; justify-content: space-between; gap: .8rem; align-items: center;}
.entity .name {font-size: 1.1rem; font-weight: 700; direction: ltr; text-align: @align;}
.entity .sub {color: @muted; font-size: .8rem;}
.ring {width: 66px; height: 66px; border-radius: 50%; flex: none; display: grid; place-items: center;}
.stApp .ring.bad {background: conic-gradient(@bad calc(var(--p) * 1%), @grid 0);}
.stApp .ring.warn {background: conic-gradient(@warn calc(var(--p) * 1%), @grid 0);}
.stApp .ring.good {background: conic-gradient(@good calc(var(--p) * 1%), @grid 0);}
.ring i {width: 52px; height: 52px; border-radius: 50%; background: @panel; color: @text; display: grid; place-items: center; font-style: normal; font-weight: 700; font-size: .98rem; direction: ltr;}
.chips {display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .7rem;}
.chip {border-radius: 8px; padding: .12rem .55rem; font-size: .76rem; background: @panel2; border: 1px solid @border; color: @text;}
.chip.ltr {direction: ltr;}
.stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin-top: .85rem; padding-top: .75rem; border-top: 1px solid @border;}
.stats div span {display: block; color: @muted; font-size: .72rem;}
.stats div b {font-size: 1.02rem; font-variant-numeric: tabular-nums;}
.action {display: grid; grid-template-columns: 38px 1fr; gap: .9rem; background: @panel; border: 1px solid @border; border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: .7rem; box-shadow: @shadow;}
.action .no {width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; font-weight: 700; background: @soft; color: @accent;}
.action .title {font-weight: 700; font-size: .98rem;}
.action .meta {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-bottom: .35rem;}
.action .ev {color: @muted; font-size: .86rem; margin-top: .25rem;}
.memo {background: linear-gradient(130deg, #052e2b, #0f766e 55%, #0369a1); border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; box-shadow: @shadow;}
.stApp .memo div, .stApp .memo span {color: #fff;}
.memo .t {font-size: 1.35rem; font-weight: 700;}
.memo .m {opacity: .85; font-size: .84rem; margin-top: .3rem;}
.memo .tag {display: inline-block; border: 1px solid rgba(255,255,255,.45); border-radius: 999px; padding: .1rem .7rem; font-size: .74rem; margin-bottom: .5rem;}
.compare .row {display: flex; align-items: baseline; gap: .6rem; direction: ltr; justify-content: @rowend;}
.compare .now {font-size: 1.9rem; font-weight: 700;}
.compare .old {color: @muted; font-size: .95rem; text-decoration: line-through; text-decoration-thickness: 1px;}
.stApp .list {margin: 0; padding: 0; list-style: none;}
.stApp .list li {display: flex; gap: .6rem; padding: .55rem 0; border-bottom: 1px solid @border; font-size: .9rem; margin: 0;}
.stApp .list li:last-child {border-bottom: 0;}
.list li::before {content: "✓"; color: @accent; font-weight: 700;}
.list.caution li::before {content: "!"; color: @warn;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin-bottom: 1rem;}
.step {background: @panel; border: 1px solid @border; border-radius: 16px; padding: .9rem; box-shadow: @shadow;}
.step i {display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 9px; background: @soft; color: @accent; font-style: normal; font-weight: 700; margin-bottom: .45rem;}
.step b {display: block; font-size: .9rem;} .step span {color: @muted; font-size: .78rem;}
.bubble {border-radius: 18px; padding: .8rem 1.05rem; margin-bottom: .55rem; max-width: 92%; width: fit-content; box-shadow: @shadow; font-size: .93rem;}
.stApp .bubble.me {background: linear-gradient(135deg, #0f766e, #0d9488); color: #fff; margin-inline-start: auto; border-end-end-radius: 5px;}
.bubble.bot {background: @panel; border: 1px solid @border; border-end-start-radius: 5px;}
.msg {border-radius: 14px; padding: .75rem 1rem; margin-bottom: .8rem; font-size: .9rem; background: @soft; color: @text;}
.stApp .msg.error {background: @badbg; color: @bad;} .stApp .msg.ok {background: @goodbg; color: @good;} .stApp .msg.warn {background: @warnbg; color: @warn;}
.tbl {background: @panel; border: 1px solid @border; border-radius: 16px; overflow: auto; max-height: 460px; margin-bottom: .9rem; box-shadow: @shadow;}
.tbl table {width: 100%; border-collapse: collapse; font-size: .85rem; margin: 0;}
.tbl th {position: sticky; top: 0; z-index: 1; background: @panel2; color: @muted; font-weight: 600; font-size: .76rem; padding: .7rem .8rem; text-align: start; white-space: nowrap; border: 0; border-bottom: 1px solid @border;}
.tbl td {padding: .62rem .8rem; border: 0; border-bottom: 1px solid @border; color: @text; white-space: nowrap; text-align: start;}
.tbl tr:last-child td {border-bottom: 0;}
.tbl td.wrap {white-space: normal; min-width: 240px;}
.tbl tbody tr:hover td {background: @panel2;}
.num {direction: ltr; display: inline-block; font-variant-numeric: tabular-nums;}
.badge {display: inline-block; border-radius: 999px; padding: .08rem .65rem; font-weight: 600; font-size: .75rem;}
.bar {display: inline-block; width: 64px; height: 6px; border-radius: 9px; background: @grid; margin-inline-start: .5rem; overflow: hidden; vertical-align: middle;}
.bar i {display: block; height: 100%; background: linear-gradient(90deg, @accent2, @accent);}
.stApp .head .t, .stApp .section, .stApp .kpi .v, .stApp .memo .t, .stApp .brand b, .stApp .entity .name, .stApp .ring i, .stApp .action .title, .stApp .compare .now, .stApp .hero .h, .stApp .feature b, .stApp h1, .stApp h2, .stApp h3 {font-family: 'Alexandria', 'Tajawal', sans-serif;}
.stApp .section small {font-family: 'Tajawal', sans-serif;}
.stApp p, .stApp li, .stApp label p {font-size: 1.02rem;}
section[data-testid="stSidebar"] [role="radiogroup"] p {font-size: 1.04rem;}
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 2.4rem 2.2rem; margin-bottom: 1.2rem; background: linear-gradient(130deg, #052e2b, #0f766e 52%, #0369a1); box-shadow: @shadow;}
.hero::after {content: ""; position: absolute; width: 380px; height: 380px; border-radius: 50%; inset-inline-end: -110px; top: -170px; background: radial-gradient(circle, rgba(251,191,36,.35), transparent 65%);}
.stApp .hero div, .stApp .hero span {color: #fff; position: relative; z-index: 1;}
.hero .tag {display: inline-block; border: 1px solid rgba(255,255,255,.4); border-radius: 999px; padding: .15rem .85rem; font-size: .82rem; margin-bottom: .9rem;}
.hero .h {font-size: 2.15rem; font-weight: 800; line-height: 1.55; max-width: 760px;}
.hero .p {font-size: 1.12rem; opacity: .92; max-width: 720px; margin-top: .6rem; line-height: 1.9;}
.feature {margin-bottom: .4rem; min-height: 200px;}
.feature i {display: inline-grid; place-items: center; width: 46px; height: 46px; border-radius: 14px; background: @soft; font-style: normal; font-size: 1.35rem; margin-bottom: .7rem;}
.feature b {display: block; font-size: 1rem; margin-bottom: .3rem;}
.stApp .feature span {display: block; color: @muted; font-size: .98rem; line-height: 1.75;}
@media (max-width: 640px) {
.hero {padding: 1.4rem 1.15rem; border-radius: 20px;} .hero .h {font-size: 1.35rem;} .hero .p {font-size: .98rem;}
.feature {min-height: 0;}
.block-container {padding: 4.25rem .75rem 2rem;}
header[data-testid="stHeader"] {background: @bg;}
.head .t {font-size: 1.45rem;} .head .s {font-size: .85rem;}
.g4 {grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem;}
.g3 {grid-template-columns: minmax(0, 1fr);}
.card {padding: .8rem .85rem;} .kpi .v {font-size: 1.45rem;} .kpi .l {font-size: .74rem;} .delta {font-size: .68rem;}
.memo {padding: 1rem;} .memo .t {font-size: 1.08rem;}
.compare .now {font-size: 1.4rem;}
.action {grid-template-columns: 30px 1fr; gap: .6rem; padding: .8rem;} .action .no {width: 28px; height: 28px;}
.chip.ltr {font-size: .7rem;}
.tbl {max-height: 380px; -webkit-overflow-scrolling: touch;} .tbl table {font-size: .76rem;} .tbl th, .tbl td {padding: .45rem .55rem;} .tbl td.wrap {min-width: 180px;} .bar {width: 40px;}
.bubble {max-width: 100%;}
.stApp input {font-size: 16px !important;}
.stButton button, .stDownloadButton button, .stFormSubmitButton button {width: 100%; min-height: 44px;}
section[data-testid="stSidebar"] [role="radiogroup"] label {min-height: 46px;}
}
</style>"""


def build_css(dark):
    rtl = lang() == "ar"
    tokens = dict(THEMES[dark], dir="rtl" if rtl else "ltr", align="right" if rtl else "left",
                  edge="-3px" if rtl else "3px", rowend="flex-end" if rtl else "flex-start")
    css = CSS
    for k in sorted(tokens, key=len, reverse=True):
        css = css.replace("@" + k, tokens[k])
    return css


def inject(dark):
    st.markdown(build_css(dark), unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def e(v):
    return escape(str(v))


def page_title(i):
    return tr(PAGES[i][0], PAGES[i][1])


def brand():
    with open("assets/logo.svg", "rb") as f:
        logo = base64.b64encode(f.read()).decode()
    st.sidebar.markdown(f'<div class="brand"><img class="logo" src="data:image/svg+xml;base64,{logo}" alt=""><div><b>'
                        f'{tr("مساعد تخطيط المرافق", "Facility Planning AI")}</b>'
                        '<small>Planning &amp; Performance Intelligence</small></div></div>', unsafe_allow_html=True)


PHOTOS = ("assets/engineer.png", "assets/engineer.jpg", "assets/engineer.jpeg", "assets/engineer.webp")


@st.cache_data(show_spinner=False)
def _photo_data(path, modified):
    """Shrink the picture so the sidebar stays fast; `modified` refreshes the cache when it changes."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((640, 900))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode()


def sidebar_art():
    """Shows assets/engineer.* if present, otherwise the drawn illustration (make_art.py)."""
    photo = next((p for p in PHOTOS if os.path.exists(p)), None)
    if photo:
        try:
            data = _photo_data(photo, os.path.getmtime(photo))
        except OSError:
            data = None  # unreadable file: fall back to the drawing
        if data:
            label = tr("مهندس تخطيط في إدارة المرافق", "Facility management planning engineer")
            st.sidebar.markdown(f'<figure class="art"><img src="data:image/jpeg;base64,{data}" alt="{label}">'
                                '</figure>', unsafe_allow_html=True)
            return
    with open("assets/sidebar.svg", encoding="utf-8") as f:
        svg = f.read()
    words = {"title": tr("إدارة المرافق الذكية", "Smart Facility Management"),
             "hvac": tr("التكييف", "HVAC"), "elec": tr("الكهرباء", "Electrical"),
             "plumb": tr("السباكة", "Plumbing"), "fire": tr("الحريق", "Fire Safety"),
             "lift": tr("المصاعد", "Elevators"), "civil": tr("المدني", "Civil"),
             "predict": tr("تنبؤ", "Predict"), "alert": tr("تنبيه", "Alert"), "decide": tr("قرار", "Decide")}
    for key, word in words.items():
        svg = svg.replace("{" + key + "}", word)
    data = base64.b64encode(svg.encode("utf-8")).decode()
    label = tr("مبنى تتصل به أنظمة التكييف والكهرباء والسباكة والحريق والمصاعد والأعمال المدنية",
               "A building connected to HVAC, electrical, plumbing, fire safety, elevator and civil systems")
    note = tr("ستة أنظمة تحت المتابعة", "Six systems under watch")
    st.sidebar.markdown(f'<figure class="art"><img src="data:image/svg+xml;base64,{data}" alt="{label}">'
                        f'<figcaption>{note}</figcaption></figure>', unsafe_allow_html=True)


def header(i, data_date):
    as_of = tr("آخر تحديث للبيانات", "Data as of")
    note = tr("بيانات اصطناعية", "Synthetic data")  # short disclosure kept on every page
    html(f'<div class="head"><div><div class="t">{page_title(i)}</div><div class="s">{tr(PAGES[i][2], PAGES[i][3])}'
         f'</div></div><span class="asof">{as_of}: <bdi>{data_date}</bdi> · {note}</span></div>')


def hero(tag, title, text):
    html(f'<div class="hero"><span class="tag">{e(tag)}</span><div class="h">{e(title)}</div>'
         f'<div class="p">{e(text)}</div></div>')


def feature(icon, title, text):
    return f'<div class="card feature"><i>{icon}</i><b>{e(title)}</b><span>{e(text)}</span></div>'


def grid(items, cols=4):
    html(f'<div class="grid g{cols}">{"".join(items)}</div>')


def kpi(label, value, delta="", tone="flat"):
    d = f'<span class="delta {tone}">{delta}</span>' if delta else '<span class="delta">&nbsp;</span>'
    return f'<div class="card kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


def insight(title, text, tone="info"):
    return f'<div class="card insight"><div class="dot {tone}"></div><div><b>{e(title)}</b><span>{e(text)}</span></div></div>'


def entity(name, sub, ring_value, tone, chips=(), stats=(), badge=""):
    """ring_value 0-100; stats: (label, value)."""
    chip = "".join(f'<span class="chip">{e(c)}</span>' for c in chips)
    cells = "".join(f"<div><span>{e(label)}</span><b>{e(v)}</b></div>" for label, v in stats)
    b = f'<span class="badge {tone}">{e(badge)}</span>' if badge else ""
    return (f'<div class="card entity"><div class="top"><div><div class="name">{e(name)}</div>'
            f'<div class="sub">{e(sub)}</div>{b}</div>'
            f'<div class="ring {tone}" style="--p:{ring_value:.0f}"><i>{ring_value:.0f}</i></div></div>'
            + (f'<div class="chips">{chip}</div>' if chip else "")
            + (f'<div class="stats">{cells}</div>' if cells else "") + "</div>")


def actions_html(recs):
    out = []
    for i, r in enumerate(recs.itertuples(), 1):
        ar, en, tone = PRIORITY[r.priority]
        ids = "".join(f'<span class="chip ltr">{e(x)}</span>' for x in r.ids)
        out.append(f'<div class="action"><div class="no">{i}</div><div><div class="meta">'
                   f'<span class="badge {tone}">{tr(ar, en)}</span>'
                   f'<span class="chip">{tr("المالك", "Owner")}: {e(r.owner)}</span></div>'
                   f'<div class="title">{e(r.action)}</div><div class="ev">{e(r.evidence)}</div>'
                   + (f'<div class="chips">{ids}</div>' if ids else "") + "</div></div>")
    return "".join(out)


def actions(recs):
    html(actions_html(recs))


def compare_card(label, now, before, delta, tone):
    return (f'<div class="card compare kpi"><div class="l">{label}</div><div class="row">'
            f'<span class="old">{before}</span><span class="now">{now}</span></div>'
            f'<span class="delta {tone}">{delta}</span></div>')


def memo_head(title, meta, tag=None):
    tag = tag or tr("مسودة تنتظر الاعتماد البشري", "Draft awaiting human approval")
    html(f'<div class="memo"><span class="tag">{e(tag)}</span><div class="t">{title}</div>'
         f'<div class="m">{meta}</div></div>')


def listing(items, caution=False):
    lis = "".join(f"<li>{e(x)}</li>" for x in items)
    return f'<ul class="list {"caution" if caution else ""}">{lis}</ul>'


def panel(title, body):
    return f'<div class="card"><div class="section" style="margin-top:0">{title}</div>{body}</div>'


def steps(items):
    html('<div class="steps">' + "".join(
        f'<div class="step"><i>{i}</i><b>{e(t)}</b><span>{e(s)}</span></div>'
        for i, (t, s) in enumerate(items, 1)) + "</div>")


def section(text, hint=""):
    html(f'<div class="section">{text}{f"<small>{hint}</small>" if hint else ""}</div>')


def message(text, kind="info"):
    html(f'<div class="msg {kind}">{e(text)}</div>')


def bubble(text, who):
    html(f'<div class="bubble {who}" dir="auto">{e(text)}</div>')


def _cell(col, v, percent):
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return f'<span class="badge bad">{tr("تنبيه", "Alert")}</span>' if v else '<span class="badge flat">—</span>'
    if pd.isna(v):
        return "—"
    if col == "risk_class" and v in RISK:
        ar, en, tone = RISK[v]
        return f'<span class="badge {tone}">{tr(ar, en)}</span>'
    if col == "probability":
        return f'<span class="num">{v:.1%}</span><span class="bar"><i style="width:{min(v * 200, 100):.0f}%"></i></span>'
    if col in percent:
        return f'<span class="num">{v:.1%}</span>'
    if col == "risk_score":
        return f'<span class="num">{v:.0f}</span><span class="bar"><i style="width:{v:.0f}%"></i></span>'
    if isinstance(v, float):
        return f'<span class="num">{v:,.0f}</span>' if float(v).is_integer() else f'<span class="num">{v:,.3f}</span>'
    if isinstance(v, int) or type(v).__name__.startswith("int"):
        return f'<span class="num">{v:,}</span>'
    return escape(str(v))


def table(frame, percent=(), index=True):
    f = frame.reset_index() if index and frame.index.name else frame
    head = "".join(f"<th>{escape(tr(*LABELS[c]) if c in LABELS else str(c))}</th>" for c in f.columns)
    rows = []
    for rec in f.to_dict("records"):
        tds = "".join(f'<td class="{"wrap" if c == "rejection_reason" else ""}">{_cell(c, v, percent)}</td>'
                      for c, v in rec.items())
        rows.append(f"<tr>{tds}</tr>")
    html(f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _style(chart, dark, height):
    t = THEMES[dark]
    return (chart.properties(height=height, background=t["panel"],
                             padding={"left": 16, "right": 16, "top": 16, "bottom": 14})
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=t["muted"], titleColor=t["muted"], gridColor=t["grid"], domain=False,
                            ticks=False, labelFontSize=12, labelPadding=8, labelLimit=220)
            .configure_legend(labelColor=t["text"], titleColor=t["muted"], orient="top"))


def _show(chart):
    st.altair_chart(chart, theme=None, width="stretch")


def trend(series, dark, height=290):
    d = series.rename("value").reset_index()
    x = d.columns[0]
    c = THEMES[dark]["accent"]
    base = alt.Chart(d).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(grid=False)),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%", tickCount=5)),
        tooltip=[alt.Tooltip(f"{x}:T", format="%Y-%m", title=tr("الشهر", "Month")),
                 alt.Tooltip("value:Q", format=".1%", title=tr("نسبة الخرق", "Breach rate"))])
    grad = alt.Gradient(gradient="linear", x1=0, x2=0, y1=1, y2=0,
                        stops=[alt.GradientStop(color=THEMES[dark]["panel"], offset=0),
                               alt.GradientStop(color=c, offset=1)])
    _show(_style(base.mark_area(opacity=.35, color=grad, interpolate="monotone") +
                 base.mark_line(color=c, strokeWidth=3, interpolate="monotone"), dark, height))


def donut(series, dark, height=290):
    d = series.rename("value").reset_index()
    x = d.columns[0]
    _show(_style(alt.Chart(d).mark_arc(innerRadius=62, outerRadius=100, cornerRadius=6, padAngle=.025).encode(
        theta=alt.Theta("value:Q"),
        color=alt.Color(f"{x}:N", title=None, scale=alt.Scale(range=SERIES),
                        legend=alt.Legend(orient="bottom", columns=2)),
        tooltip=[f"{x}:N", alt.Tooltip("value:Q", format=",")]), dark, height))


def lines(frame, dark):
    d = frame.reset_index().melt(id_vars=frame.index.name, var_name="series", value_name="value")
    _show(_style(alt.Chart(d).mark_line(strokeWidth=2.5, interpolate="monotone").encode(
        x=alt.X(f"{frame.index.name}:T", title=None, axis=alt.Axis(grid=False)),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=SERIES)),
        tooltip=["series:N", alt.Tooltip("value:Q", format=".1%")]), dark, 320))


def bars(series, dark, fmt="%"):
    d = series.rename("value").reset_index()
    x = d.columns[0]
    _show(_style(alt.Chart(d).mark_bar(cornerRadius=7, height={"band": .62}, color=THEMES[dark]["accent"]).encode(
        y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X("value:Q", title=None, axis=alt.Axis(format=fmt)),
        tooltip=[f"{x}:N", alt.Tooltip("value:Q", format=".1%" if fmt == "%" else ".4f")]),
        dark, max(170, 38 * len(d))))


def grouped_bars(frame, dark):
    d = frame.reset_index().melt(id_vars=frame.index.name, var_name="series", value_name="value")
    d["series"] = d.series.map({"predicted": tr("المتوقع", "Predicted"), "actual": tr("الفعلي", "Actual")})
    _show(_style(alt.Chart(d).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X(f"{frame.index.name}:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="series:N",
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=SERIES)),
        tooltip=["series:N", alt.Tooltip("value:Q", format=".1%")]), dark, 280))


def memo_document(title, meta, compare_cards, recs, caveats, reviewer):
    """Standalone printable HTML of the approved memo (light theme, current language)."""
    rtl = lang() == "ar"
    body = (f'<div class="memo"><span class="tag">{tr("معتمدة من", "Approved by")}: {e(reviewer)}</span>'
            f'<div class="t">{title}</div><div class="m">{meta}</div></div>'
            f'<div class="grid g3">{"".join(compare_cards)}</div>'
            f'<div class="section">{tr("الإجراءات المقترحة", "Proposed actions")}</div>{actions_html(recs)}'
            f'<div class="section">{tr("تحفظات", "Caveats")}</div><div class="card">{listing(caveats, True)}</div>')
    return (f'<!doctype html><html lang="{lang()}" dir="{"rtl" if rtl else "ltr"}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{title}</title>{build_css(False)}<style>body{{font-family:'Tajawal',Tahoma,"
            "sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#101828}.action,.card"
            "{break-inside:avoid}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}</style></head>"
            f'<body class="stApp">{body}</body></html>')
