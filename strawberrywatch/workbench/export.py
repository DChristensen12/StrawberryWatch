"""CSV exports and the 1-page calibration summary PDF."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from strawberrywatch.workbench import text

TITLE = "StrawberryWatch calibration summary"
LABELS = ("Model", "Saved", "Data", "Left out", "False alarms", "Notes")
TABLE_TITLE = "Past events"
COLUMNS = ("Event", "Should a model alert?", "Now", "After recalibrating")
FOOTER = "Made with the StrawberryWatch Workbench on {date}."


def csv_text(header, rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def summary_pdf(fields, events, today=None):
    """fields are the 6 values in LABELS order, events are (name, should_alert, now, after) rows."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    today = today or datetime.now(text.LOCAL).date()
    out = io.BytesIO()
    plt.rcParams["font.family"] = "DejaVu Sans"
    # 1 US Letter page, in inches
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.94, TITLE, fontsize=18, weight="bold", va="top")
    y = 0.89
    for label, value in zip(LABELS, fields, strict=True):
        fig.text(0.08, y, label, fontsize=11, weight="bold", va="top")
        fig.text(0.30, y, value or "", fontsize=11, va="top", wrap=True)
        y -= 0.035
    y -= 0.02
    fig.text(0.08, y, TABLE_TITLE, fontsize=11, weight="bold", va="top")
    y -= 0.02
    if events:
        ax = fig.add_axes(
            [0.08, max(0.08, y - 0.03 * (len(events) + 1)), 0.84, 0.03 * (len(events) + 1)]
        )
        ax.axis("off")
        table = ax.table(
            cellText=[list(r) for r in events], colLabels=COLUMNS, loc="upper left", cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
    fig.text(0.08, 0.04, FOOTER.format(date=text.long_dates(today)), fontsize=9, va="bottom")
    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return out.getvalue()
