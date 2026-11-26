"""
i18n.py - Arabic / English switch. Each Streamlit session runs in its own context,
so one user's language never changes another's.
"""
from contextvars import ContextVar

_LANG = ContextVar("lang", default="ar")


def set_lang(code):
    _LANG.set("en" if code == "en" else "ar")


def lang():
    return _LANG.get()


def tr(ar, en):
    return en if _LANG.get() == "en" else ar
