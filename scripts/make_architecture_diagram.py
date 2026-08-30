#!/usr/bin/env python3
"""Render docs/architecture.{svg,png} for the hackathon submission (M7)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a2733"
MUT = "#5d6f7f"
LINE = "#b9c4cd"
ACCENT = "#0b5fa5"
OK = "#1c7c46"
WARN = "#b3610e"
BAD = "#a52a2a"
PUB = "#e8f1f8"
PRIV = "#fdf3e7"
DATA = "#eef7ee"
NEUTRAL = "#f4f6f8"

fig, ax = plt.subplots(figsize=(13.5, 10.5), dpi=160)
ax.set_xlim(0, 135)
ax.set_ylim(0, 105)
ax.axis("off")


def box(x, y, w, h, title, sub="", fc=NEUTRAL, ec=LINE, title_c=INK, fs=10.5, sfs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=1.4))
    cy = y + h / 2 + (2.2 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=fs,
            color=title_c, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                fontsize=sfs, color=MUT)
    return (x, y, w, h)


def arrow(p1, p2, label="", color=INK, lw=1.6, ls="-", label_dx=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_dx, my, label, fontsize=7.8, color=MUT, ha="left", va="center")


def cx(b):  # bottom-center / top-center helpers
    return b[0] + b[2] / 2


# ── title ────────────────────────────────────────────────────────────────────
ax.text(2, 102.5, "TopoScout — autonomous structural verification for scientific imaging AI",
        fontsize=15, color=INK, fontweight="bold")
ax.text(2, 99.6, "Gemini coordinates · deterministic tools measure · topology controls the action · bounded policy decides",
        fontsize=9.5, color=MUT)

# ── main pipeline (left column) ──────────────────────────────────────────────
user = box(24, 92, 34, 5.5, "USER / SCIENTIST", fc="#ffffff")
ui = box(24, 82.5, 34, 6.5, "TopoScout Web UI", "Cloud Run · PUBLIC", fc=PUB, ec=ACCENT)
gcs = box(4, 70.6, 24, 7.6, "Cloud Storage", "input · masks · overlays\nprob maps · reports", fc=DATA, ec=OK)
fs_ = box(56, 70.6, 24, 7.6, "Firestore", "exact run state +\nverbatim tool evidence", fc=DATA, ec=OK)
tasks = box(30, 71.5, 22, 6.5, "Cloud Tasks", "{run_id, image_uri}", fc=PUB, ec=ACCENT)
agent = box(24, 60.5, 34, 7.5, "Gemini 3.5 Flash Agent", "Google ADK · Cloud Run · PRIVATE\nchooses only WHEN to call approved tools", fc=PRIV, ec=WARN)
qc = box(24, 52.5, 34, 5.2, "Image Quality Control", "deterministic QC on the canonical input", fc=NEUTRAL)
worker = box(24, 42.5, 34, 7.2, "Scientific Model Worker", "private Cloud Run · DINOv2 T0 segmenter\nfixed checkpoint · frozen thresholds", fc=PRIV, ec=WARN)
audit = box(24, 32.5, 34, 7.2, "TOPOLOGICAL AUDIT", "β₀ · β₁ · fragmentation · tiny components\nlargest component fraction", fc="#f0ebf7", ec="#6a4fa3")
policy = box(24, 23.5, 34, 6.2, "BOUNDED POLICY", "sole decision authority", fc=NEUTRAL, ec=INK)

accept = box(6, 13, 20, 5.5, "ACCEPT", fc="#e8f5ec", ec=OK, title_c=OK)
retry = box(31, 13, 20, 5.5, "RETRY (once)", fc="#fdf1e2", ec=WARN, title_c=WARN)
review = box(56, 13, 22, 5.5, "HUMAN REVIEW", fc="#faeaea", ec=BAD, title_c=BAD)
recover = box(31, 4, 20, 5.5, "validated recovery", "500 px significance filter", fc=NEUTRAL)

evidence = box(4, 4, 24, 5.5, "evidence report", "Firestore + GCS", fc=DATA, ec=OK)

arrow((cx(user), 92), (cx(ui), 89), "upload image")
arrow((30, 82.5), (16, 78), "original", color=OK)
arrow((52, 82.5), (66, 78), "run doc", color=OK)
arrow((cx(ui), 82.5), (cx(tasks), 78), "202 + enqueue")
arrow((cx(tasks), 71.5), (cx(agent), 68), "OIDC-authenticated")
arrow((cx(agent), 60.5), (cx(qc), 57.7))
arrow((cx(qc), 52.5), (cx(worker), 49.7))
arrow((cx(worker), 42.5), (cx(audit), 39.7), "mask")
arrow((cx(audit), 32.5), (cx(policy), 29.7))
arrow((30, 23.5), (16, 18.5), color=OK)
arrow((cx(policy), 23.5), (41, 18.5), color=WARN)
arrow((52, 23.5), (67, 18.5), color=BAD)
arrow((41, 13), (41, 9.5), color=WARN)
arrow((31, 6.8), (28.6, 34), "re-audit", color=WARN, ls="--", lw=1.3, label_dx=1.0)
arrow((16, 13), (16, 9.5), color=OK)
arrow((67, 13), (28, 7.5), color=OK, ls="--", lw=1.2)

# evidence writes back up to Firestore/GCS
arrow((10, 9.5), (10, 71.5), color=OK, ls=":", lw=1.1, label_dx=-9)
ax.text(1.2, 40, "every tool output persisted verbatim", fontsize=7.6, color=OK, rotation=90, va="center")

# ── adapter panel (right column) ─────────────────────────────────────────────
panel_x = 86
ax.add_patch(FancyBboxPatch((panel_x - 2, 2), 49, 92, boxstyle="round,pad=0.8,rounding_size=1.6",
                            facecolor="#fbfcfd", edgecolor=LINE, linewidth=1.2, linestyle="--"))
ax.text(panel_x + 22.5, 91.2, "SAME TOPOSCOUT CONTRACT", ha="center", fontsize=11,
        color=INK, fontweight="bold")
ax.text(panel_x + 22.5, 88.4, "registered adapters only — the LLM never picks\nmodels, checkpoints, thresholds, or paths",
        ha="center", fontsize=8, color=MUT)

ref = box(panel_x + 2, 78, 41, 7, "Maize lesion model", "VALIDATED REFERENCE APPLICATION\nRETRY→ACCEPT · RETRY→HUMAN_REVIEW", fc="#e8f5ec", ec=OK)
pilots = [
    ("Cellpose — Microscopy", "40 instances · plausible"),
    ("HoVer-Net — Pathology", "8 nuclei · plausible · NOT clinical"),
    ("Road U-Net — Satellite", "26 disconnected pieces → suspicious"),
    ("CrackenPy — Materials", "335 crack fragments → suspicious"),
    ("PatchCore — Industrial", "β₀ = 1 coherent defect region"),
]
y = 68.5
pboxes = []
for name, res in pilots:
    pboxes.append(box(panel_x + 2, y, 41, 6.6, name, res + "\nPORTABILITY PILOT — not domain-validated",
                      fc="#fdf9f2", ec=WARN, fs=9.3, sfs=7.4))
    y -= 9.2

paudit = box(panel_x + 7, 14, 31, 6.5, "structural audit", "domain profiles select the metrics", fc="#f0ebf7", ec="#6a4fa3")
for b in [ref] + pboxes:
    arrow((panel_x + 1, b[1] + b[3] / 2), (panel_x - 0.5, b[1] + b[3] / 2), color=MUT, lw=1.0)
    arrow((cx(b), b[1]), (cx(paudit), 20.5), color="#6a4fa3", lw=0.9, ls=":")

ax.text(panel_x - 4.5, 50, "adapter API", fontsize=8, color=MUT, rotation=90, va="center")

# legend
lx = 86.5
ax.add_patch(FancyBboxPatch((lx, 3.2), 20, 2.6, boxstyle="round,pad=0.3", facecolor=PUB, edgecolor=ACCENT, lw=1))
ax.text(lx + 22, 4.5, "public service", fontsize=8, color=MUT, va="center")
ax.add_patch(FancyBboxPatch((lx + 33.5, 3.2), 8, 2.6, boxstyle="round,pad=0.3", facecolor=PRIV, edgecolor=WARN, lw=1))
ax.text(lx + 43, 4.5, "private", fontsize=8, color=MUT, va="center")

fig.savefig("docs/architecture.svg", bbox_inches="tight", facecolor="white")
fig.savefig("docs/architecture.png", bbox_inches="tight", facecolor="white")
print("wrote docs/architecture.svg + docs/architecture.png")
