#!/usr/bin/env python3
"""Render docs/architecture.{svg,png} — simple, GitHub-friendly (M7)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2937"
MUT = "#6b7280"
LINE = "#9aa7b1"
DARK = "#374151"
OK = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"

fig, ax = plt.subplots(figsize=(11.5, 10.2), dpi=180)
ax.set_xlim(0, 115)
ax.set_ylim(0, 102)
ax.axis("off")


def box(x, y, w, h, title, sub="", ec=LINE, fc="#ffffff", tc=INK, fs=12, sfs=8.6, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    if sub:
        ax.text(x + w / 2, y + h / 2 + 1.5, title, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center", va="center",
                fontsize=sfs, color=MUT)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold")
    return (x, y, w, h)


def a(p1, p2, color=DARK, lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2.5, shrinkB=2.5))


def top(b):   return (b[0] + b[2] / 2, b[1] + b[3])
def bot(b):   return (b[0] + b[2] / 2, b[1])
def left(b):  return (b[0], b[1] + b[3] / 2)
def right(b): return (b[0] + b[2], b[1] + b[3] / 2)


ax.text(57.5, 99.3, "TopoScout", fontsize=19, color=INK, fontweight="bold", ha="center")
ax.text(57.5, 95.9, "AI models make predictions. TopoScout checks whether they should be trusted.",
        fontsize=10, color=MUT, ha="center")

X, W, H = 20, 42, 7.2
ui = box(X, 85, W, H, "Web UI", "Cloud Run · public")
tasks = box(X, 74.8, W, H, "Cloud Tasks", "async, reliable · {run_id, image_uri}")
agent = box(X, 64.6, W, H, "Gemini Agent", "Google ADK · Cloud Run · private", ec=DARK, lw=1.9)
model = box(X, 54.4, W, H, "Scientific Model", "Cloud Run · private · fixed checkpoint")
audit = box(X, 44.2, W, H, "Topology Audit", "β₀ · β₁ · fragmentation", ec=DARK, lw=1.9)
policy = box(X, 34, W, H, "Bounded Policy", "deterministic decision", ec=DARK, lw=1.9)

accept = box(4, 20, 26, 7.2, "ACCEPT", ec=OK, tc=OK, fc="#f2faf4")
retry = box(36, 20, 26, 7.2, "RETRY once", "validated recovery", ec=WARN, tc=WARN, fc="#fdf7ec")
review = box(68, 20, 26, 7.2, "HUMAN REVIEW", ec=BAD, tc=BAD, fc="#fcf1f1")

store = box(74, 54.4, 36, 17.6, "", "", ec=LINE)
ax.text(92, 66.5, "Evidence store", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.text(92, 63.6, "Firestore + Cloud Storage", ha="center", fontsize=9, color=INK)
ax.text(92, 58.2, "every tool output saved verbatim —\nthe agent can never invent\nor round a measurement",
        ha="center", fontsize=8.4, color=MUT)

a(bot(ui), top(tasks))
a(bot(tasks), top(agent))
a(bot(agent), top(model))
a(bot(model), top(audit))
a(bot(audit), top(policy))
a(left(policy), top(accept), color=OK, rad=0.25)
a(bot(policy), top(retry), color=WARN)
a(right(policy), top(review), color=BAD, rad=-0.25)
a(right(retry), (right(audit)[0] + 3, right(audit)[1] - 2), color=WARN, ls="--", lw=1.6, rad=-0.35)
ax.text(70.5, 38.5, "re-audit", fontsize=8.6, color=WARN, ha="center")
a(right(agent), (74, 63), color=MUT, ls=":", lw=1.4, rad=-0.12)
ax.text(69.5, 69, "evidence", fontsize=8.6, color=MUT, ha="center")

ax.text(57.5, 14.9, "upload → model → structural check → act → evidence", fontsize=9,
        color=MUT, ha="center", style="italic")

band = box(4, 2.5, 106, 8.6, "", ec=LINE)
ax.text(57, 8.9, "One contract, many domains", ha="center", fontsize=10.5,
        color=INK, fontweight="bold")
ax.text(57, 5.2,
        "Maize lesion model (VALIDATED)   ·   Cellpose   ·   HoVer-Net   ·   Road U-Net   ·   CrackenPy   ·   PatchCore  (portability pilots)",
        ha="center", fontsize=8.8, color=MUT)

fig.savefig("docs/architecture.svg", bbox_inches="tight", facecolor="white")
fig.savefig("docs/architecture.png", bbox_inches="tight", facecolor="white")
print("wrote docs/architecture.svg + docs/architecture.png")
