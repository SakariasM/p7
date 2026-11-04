"""
The graphs for the report from usage_report.py: per day, LLM calls per question (left axis)
against accuracy (right axis), with the share of questions answered from the learned-question
store as bars. One figure for the nightly benchmark (accuracy on the fixed eval sets) and one
for the live questions (accuracy = share of the map's verdicts that were "correct").
Needs matplotlib.

    cd harness
    python3 usage_plot.py     # logs/usage_benchmark.{pdf,png}, logs/usage_live.{pdf,png}
"""

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import usage_report  # noqa: E402


def plot(rows: list[dict], accuracy_key: str, accuracy_label: str, title: str,
         name: str) -> None:
    if not rows:
        print(f"{name}: no data yet")
        return
    days = [date.fromisoformat(r["day"]) for r in rows]
    # Two panels on one time axis: LLM calls on their own scale, the percentages together.
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 4.8),
                                      gridspec_kw={"height_ratios": [1, 1.2]})
    top.plot(days, [r["llm_calls_per_question"] for r in rows], "o-", color="#1f5fbf")
    top.set_ylabel("LLM calls\nper question")
    top.set_ylim(bottom=0)
    top.set_title(title)
    top.grid(axis="y", alpha=0.3)
    bottom.bar(days, [100 * (r["from_store"] or 0) for r in rows], width=0.6, color="#c9d6ea",
               label="answered from the store")
    bottom.plot(days, [None if r[accuracy_key] is None else 100 * r[accuracy_key]
                       for r in rows], "s-", color="#b42318", label=accuracy_label)
    bottom.set_ylabel("%")
    bottom.set_ylim(0, 105)
    bottom.grid(axis="y", alpha=0.3)
    bottom.legend(loc="center left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(usage_report.REPORT_DIR / f"usage_{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"{name}: {len(rows)} days -> {usage_report.REPORT_DIR}/usage_{name}.pdf/.png")


if __name__ == "__main__":
    plot(usage_report.benchmark(), "accuracy", "accuracy",
         "Nightly benchmark (fixed eval sets)", "benchmark")
    plot(usage_report.live(), "correct_share", "answers marked correct",
         "Questions asked on the map", "live")
