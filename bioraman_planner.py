"""
BioRaman — Analysis Planner (offline)
=====================================
An in-app assistant that ASKS the user about their sample and goal, then SUGGESTS
a concrete, step-by-step BioRaman workflow — entirely from a built-in knowledge
base. No internet, no external search.

Hook into bioraman.py:
    import bioraman_planner as bp
    am.add_command(label="🧭  Analysis Planner…", command=lambda: bp.open_planner(self))

Author: Akalabya Bissoyi  •  github.com/Akalabyabissoyi/BioRaman
"""
from __future__ import annotations

# ---- knowledge base: goal -> ordered workflow steps (BioRaman feature names) ----
GOALS = {
    "Compare conditions / cryoprotectants (ice suppression)": [
        "Load every map (File → Open Raman file) — keep one file per field/cell.",
        "Preprocessing → Settings: use the default pipeline (cosmic-ray, dark, AsLS "
        "baseline, Savitzky–Golay, area normalise). Export the recipe (JSON) once and "
        "reuse it so all maps are treated identically.",
        "Define the ice/liquid bands: A = 3085–3165 cm⁻¹ (ordered/ice OH), "
        "B = 3350–3550 cm⁻¹ (disordered/liquid OH).",
        "Draw an ROI on each cell (sidebar → Draw ROI) or let auto-segmentation find it.",
        "Analysis → Group Comparison: add each condition as a group, set the A/B bands, "
        "Preview cell masks, then Run comparison.",
        "Read the Box plot + stats tab (Kruskal–Wallis for ≥3 groups, Mann–Whitney for 2, "
        "with Hodges–Lehmann CI and Cliff's δ), the Mean spectra tab, and the A/B maps tab.",
        "Export figures. For publication, report n cells AND independent experiments.",
    ],
    "Quantify intracellular ice in one map": [
        "Load the map and apply the default preprocessing.",
        "Use the ice OH band 3087–3162 cm⁻¹ (Yu et al.) with side-of-peak background.",
        "Draw an ROI around the cell (sidebar → Draw ROI).",
        "Analysis → ROI Analysis: set Band A (cell/CH) and Band B (ice), binarise, and read "
        "the per-pixel maps, mean spectrum, and pixel counts.",
        "Report the ice/liquid ratio and/or % ice-positive pixels within the ROI.",
    ],
    "Identify the chemical components in a map": [
        "Load the map and preprocess (default recipe).",
        "Analysis → MCR-ALS: choose the number of components (start with 2–4), run to "
        "convergence; inspect recovered pure spectra + abundance maps.",
        "If endmembers are expected to be the purest pixels, try Analysis → N-FINDR instead.",
        "Confirm each component: Analysis → Library Search (full-spectrum) against your "
        "reference library, or Peak Identification for band-by-band assignment.",
        "Report % lack-of-fit / explained variance and justify the component count.",
    ],
    "Map a specific molecule or bond": [
        "Load and preprocess the map.",
        "Identify the characteristic band (Peak Identification if unsure).",
        "Use the ÷ Ratio / band tools to make a peak-intensity or band-ratio map at that "
        "wavenumber; adjust Band A/B and the colour limits.",
        "Optionally overlay on the white-light image for context.",
    ],
    "Cluster / segment regions of a map": [
        "Load and preprocess the map.",
        "Analysis → Cluster Analysis: pick K-means or hierarchical, set 2–10 clusters.",
        "Check the mean silhouette score to judge the cluster count; inspect per-cluster "
        "mean spectra and the colour-coded cluster map.",
        "Export the cluster map (PNG) and label matrix (CSV).",
    ],
    "Identify an unknown spectrum": [
        "Load the data and preprocess.",
        "Select the spectrum/ROI of interest.",
        "Analysis → Peak Identification to detect and label peaks against the band library.",
        "Analysis → Library Search (full-spectrum) against RRUFF / Raman Open Database / "
        "SLoPP for a ranked match list.",
    ],
    "Measure particle size / count": [
        "Load and preprocess the map.",
        "Analysis → Particle Statistics: Otsu binarisation gives counts, equivalent circle "
        "diameter, area %, a histogram and a CSV export.",
    ],
    "Analyse depth / 3D volume": [
        "Acquire or load a Z-stack (one WDF per depth plane), or use a 2-D map with a "
        "synthetic depth axis.",
        "Analysis → 3D Volume Viewer: choose Volume Scatter, Orthogonal Slices, 3D Surface, "
        "or Multi-band RGB; set threshold, σ, Z-scale and colour map; export at 250 dpi.",
    ],
    "Quantify components against reference spectra": [
        "Load and preprocess the map.",
        "Analysis → Component Analysis (DCLS/NNLS): load your reference spectra to get "
        "concentration maps, % lack-of-fit and concentration estimates.",
    ],
    "Explore variation / reduce dimensionality": [
        "Load one or many maps and preprocess identically.",
        "Analysis → PCA Analysis (or PCA Studio for large datasets): inspect scores, "
        "loadings and the Hotelling-T² / Q diagnostics.",
        "Interpret loadings against known bands before any classification.",
    ],
}

# free-text keywords -> goal key (offline matching)
KEYWORDS = {
    "Compare conditions / cryoprotectants (ice suppression)":
        ["compare", "condition", "cryoprotect", "dmso", "polyampholyte", "jk100", "cpa",
         "control", "group", "freeze", "cryo", "ice suppress", "vitrif"],
    "Quantify intracellular ice in one map":
        ["ice", "intracellular", "quantify ice", "oh", "water", "frozen cell"],
    "Identify the chemical components in a map":
        ["component", "unmix", "pure spectra", "mcr", "endmember", "nfindr", "resolve"],
    "Map a specific molecule or bond":
        ["map a", "molecule", "bond", "band", "peak intensity", "lipid map", "protein map"],
    "Cluster / segment regions of a map":
        ["cluster", "segment", "region", "kmeans", "k-means", "hierarchical"],
    "Identify an unknown spectrum":
        ["unknown", "identify", "what is", "match", "library", "database"],
    "Measure particle size / count":
        ["particle", "size", "count", "diameter", "grain"],
    "Analyse depth / 3D volume":
        ["3d", "depth", "volume", "z-stack", "confocal", "slice"],
    "Quantify components against reference spectra":
        ["reference", "cls", "dcls", "nnls", "concentration"],
    "Explore variation / reduce dimensionality":
        ["pca", "variance", "dimensionality", "score", "loading", "outlier"],
}

SAMPLE_NOTES = {
    "Frozen cells (cryo-Raman)":
        "Keep the cryo-stage at the target temperature and allow 5–10 min equilibration "
        "before imaging; acquire brightfield + Raman together.",
    "Live/fixed cells (room temp)":
        "Watch for laser-induced damage; use the lowest power that gives adequate SNR.",
    "Tissue section":
        "Large maps benefit from PCA Studio (scalable) and batch preprocessing.",
    "Particles / powder":
        "Particle Statistics needs a clean binarisation — check the Otsu threshold.",
    "Mineral / geological":
        "Full-spectrum Library Search (RRUFF) is usually the fastest identification route.",
    "Other / not sure":
        "",
}


import os as _os, json as _json
_CFG = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".bioraman_planner.json")

def _load_cfg():
    try:
        return _json.load(open(_CFG))
    except Exception:
        return {}

def _save_cfg(d):
    try:
        _json.dump(d, open(_CFG, "w"))
    except Exception:
        pass

def should_show_startup():
    """True if the planner should auto-open at GUI launch (default yes)."""
    return bool(_load_cfg().get("show_startup", True))


def suggest_from_text(text):
    """Return the best-matching goal key from free text, or None."""
    t = (text or "").lower()
    best, score = None, 0
    for goal, kws in KEYWORDS.items():
        s = sum(1 for k in kws if k in t)
        if s > score:
            best, score = goal, s
    return best if score else None


def build_plan(goal, sample):
    lines = [f"ANALYSIS PLAN — {goal}", "=" * 60, ""]
    note = SAMPLE_NOTES.get(sample, "")
    if note:
        lines += [f"Sample note ({sample}): {note}", ""]
    for i, step in enumerate(GOALS[goal], 1):
        lines.append(f"{i}. {step}")
    lines += ["",
              "General reminders:",
              " • Apply ONE saved preprocessing recipe to every map for a fair comparison.",
              " • Report the exact band windows, n (cells and experiments), and the test used.",
              " • Use the Map display options (smooth / equal size) for publication figures.",
              " • Everything above runs inside BioRaman — no external tools required."]
    return "\n".join(lines)


def open_planner(parent=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("BioRaman — Analysis Planner")
    win.geometry("820x600")

    top = ttk.Frame(win, padding=12); top.pack(fill="x")
    ttk.Label(top, text="Tell me about your analysis and I'll suggest a BioRaman workflow.",
              font=("", 11, "bold")).pack(anchor="w", pady=(0, 8))

    r1 = ttk.Frame(top); r1.pack(fill="x", pady=3)
    ttk.Label(r1, text="Sample type:", width=16).pack(side="left")
    sample = tk.StringVar(value="Other / not sure")
    ttk.Combobox(r1, textvariable=sample, values=list(SAMPLE_NOTES), state="readonly",
                 width=40).pack(side="left")

    PLACEHOLDER = "— Select a goal —"
    r2 = ttk.Frame(top); r2.pack(fill="x", pady=3)
    ttk.Label(r2, text="Primary goal:", width=16).pack(side="left")
    goal = tk.StringVar(value=PLACEHOLDER)
    ttk.Combobox(r2, textvariable=goal, values=[PLACEHOLDER] + list(GOALS), state="readonly",
                 width=52).pack(side="left")

    r3 = ttk.Frame(top); r3.pack(fill="x", pady=3)
    ttk.Label(r3, text="…or describe it:", width=16).pack(side="left")
    free = tk.StringVar()
    ttk.Entry(r3, textvariable=free, width=54).pack(side="left")
    ttk.Label(top, text="(Free text is matched to the closest goal — offline, no search.)",
              foreground="#666").pack(anchor="w")

    out = tk.Text(win, wrap="word", height=20, font=("", 11))
    out.pack(fill="both", expand=True, padx=12, pady=(6, 6))

    def generate():
        g = goal.get()
        if free.get().strip():
            m = suggest_from_text(free.get())
            if m:
                g = m; goal.set(m)
        out.delete("1.0", "end")
        if g not in GOALS:
            out.insert("1.0",
                       "Choose a sample type and a primary goal above (or describe your "
                       "goal in the box), then click ‘Generate plan’.\n\n"
                       "BioRaman is a general-purpose Raman map analysis tool — this "
                       "planner suggests a workflow for whatever analysis you select; it "
                       "is not specific to any single assay.")
            return
        out.insert("1.0", build_plan(g, sample.get()))

    def save():
        txt = out.get("1.0", "end").strip()
        if not txt:
            messagebox.showinfo("Nothing to save", "Generate a plan first."); return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Text", "*.txt")], title="Save plan")
        if p:
            open(p, "w", encoding="utf-8").write(txt)
            messagebox.showinfo("Saved", "Plan saved to\n" + p)

    bar = ttk.Frame(win, padding=(12, 0, 12, 12)); bar.pack(fill="x")
    ttk.Button(bar, text="🧭 Generate plan", command=generate).pack(side="left")
    ttk.Button(bar, text="⬇ Save plan…", command=save).pack(side="left", padx=6)
    startup_var = tk.BooleanVar(value=should_show_startup())
    def _toggle():
        d = _load_cfg(); d["show_startup"] = startup_var.get(); _save_cfg(d)
    ttk.Checkbutton(bar, text="Show at startup", variable=startup_var,
                    command=_toggle).pack(side="right")
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right", padx=6)
    generate()
    return win


if __name__ == "__main__":
    open_planner()
