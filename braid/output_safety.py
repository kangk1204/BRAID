"""Output hardening shared by BRAID's table writers (TSV / Excel).

Gene names and event ids flow from upstream caller tables into BRAID's output
files. A field that starts with ``=``, ``+``, ``-``, ``@`` (or a control character)
is interpreted as a *formula* when the TSV/CSV/XLSX is opened in Excel or
LibreOffice — a CSV/spreadsheet formula-injection vector if the upstream table was
attacker-influenced. We neutralise it at write time.
"""

from __future__ import annotations

_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def csv_safe(text: str) -> str:
    """Return *text* with spreadsheet formula injection neutralised.

    A value whose first character can start a formula is prefixed with an apostrophe
    so Excel/LibreOffice render it as inert text. A value that *already* starts with
    an apostrophe is also guarded (doubled), so that any apostrophe-leading value in a
    BRAID-written file was guarded and :func:`csv_restore` is an exact inverse. Values
    that do not begin with a trigger (numbers, normal gene/event ids) are unchanged.
    """
    if text and (text[0] in _FORMULA_LEADERS or text[0] == "'"):
        return "'" + text
    return text


def csv_restore(text: str) -> str:
    """Inverse of :func:`csv_safe`: strip exactly one guard apostrophe.

    Because :func:`csv_safe` guards every value that begins with a formula leader OR
    an apostrophe, any apostrophe-leading value BRAID wrote was guarded, so removing a
    single leading apostrophe recovers the original exactly — a true round-trip for
    machine-read TSVs keyed/joined on event or gene ids.
    """
    if text.startswith("'"):
        return text[1:]
    return text
