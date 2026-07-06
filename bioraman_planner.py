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

# Developer contact shown in the planner (institutional address; ~48 h response).
DEV_EMAIL = "akalabya.bissoyi@manchester.ac.uk"

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
    "PCA / explore variation / classify (PCA Studio)": [
        "Load one or many maps/spectra and preprocess them with the SAME saved recipe "
        "so differences are chemical, not processing artefacts.",
        "Choose a normalisation appropriate to PCA (e.g. SNV or vector) and mean-centre; "
        "PCA Studio applies this before decomposition.",
        "Analysis → PCA Analysis, or Analysis → PCA Studio for large datasets / more "
        "controls (density-cloud view, subset re-analysis, robust PCA for outlier-heavy "
        "data).",
        "Inspect the scree/explained-variance to pick the number of PCs; read the score "
        "plot for grouping and the loadings to see which bands drive the separation.",
        "Interpret loadings against known Raman bands BEFORE trusting any grouping "
        "(physics-informed check).",
        "Use Hotelling-T² and Q-residual (SPE) to flag outliers / distance-to-reference; "
        "these also work as a simple stability/quality score.",
        "For supervised classification, run PLS-DA on the scores and report "
        "cross-validated accuracy as a baseline. Export score/loading plots (PNG/PDF).",
    ],
    "Line profile / line scan across a map": [
        "Load the map and apply the default preprocessing.",
        "Analysis → Line Profile (line scan): set Band 1 (and optionally Band 2 for a "
        "dual axis, e.g. one component vs another).",
        "Click two points on the map to draw the line across the feature (for example "
        "from the extracellular space through the middle of a cell/nucleus).",
        "Plot profile: band intensity versus distance along the line (in µm, calibrated "
        "from the map's pixel size). If you drew an ROI, shade the intracellular region.",
        "Export the CSV (distance, band 1, band 2) and the figure. Confirm the band "
        "windows match the peaks in your own spectra.",
    ],
    "Preprocess data / build a reproducible pipeline": [
        "Load a map or spectrum.",
        "Preprocessing → Settings: set cosmic-ray removal, dark/pedestal subtraction, "
        "baseline correction (AsLS), Savitzky–Golay smoothing and normalisation.",
        "Preprocessing → Export Pipeline (JSON) to save the recipe; reuse it on other "
        "files so every dataset is treated identically.",
        "Preprocessing → Reprocess to re-apply a changed recipe to the retained raw data; "
        "use Processing Log to record the exact parameters for your methods section.",
    ],
    "Check data quality (SNR / intensity / saturation)": [
        "Load the map.",
        "Preprocessing → Quality Control Maps: view per-pixel SNR, total/maximum "
        "intensity and detector-saturation maps.",
        "Use these to exclude saturated or low-SNR pixels before quantitative analysis.",
    ],
    "Batch-process a folder of maps": [
        "Preprocessing → Export Pipeline to save one recipe (JSON).",
        "File → Batch Process Folder: apply that recipe to every file in a folder; a "
        "batch_summary.csv is written alongside the processed outputs.",
        "For servers/pipelines, run headless: python bioraman.py --input <folder> "
        "--out <folder> [--recipe recipe.json].",
    ],
    "Compare or combine several maps (multi-map)": [
        "Load and preprocess each map with the same recipe.",
        "Analysis → Multi-Map Analysis to view/compare maps side by side or pooled.",
        "For per-group statistics use Analysis → Group Comparison; for variation across "
        "maps use PCA Studio.",
    ],
    "Edit a map (resample / crop / rotate / subtract background)": [
        "Load the map.",
        "Analysis → Spectral Tools: resample to an equidistant wavenumber grid, crop the "
        "map to a pixel bounding box, rotate 0/90/180/270°, or subtract an optical "
        "substrate/background reference.",
        "Save the processed data (File → Save Processed Data) to keep the edited cube.",
    ],
    "Extract endmembers / novel regions (N-FINDR, FlyHash)": [
        "Load and preprocess the map.",
        "Analysis → N-FINDR: find the purest spectral signatures (endmembers) and their "
        "NNLS abundance maps.",
        "Analysis → FlyHash (Region + Anomaly): detect distinct regions and flag "
        "anomalous/novel pixels.",
    ],
    "Save a report or session": [
        "After analysis, File → Save Analysis Report (HTML) for a self-contained record "
        "(map, mean spectrum, recipe and log).",
        "File → Save Session to store the full state; Load Session to resume later.",
        "Individual windows also export their plots as PNG/PDF and tables as CSV.",
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
    "Line profile / line scan across a map":
        ["line profile", "line scan", "linescan", "line-scan", "transect", "profile",
         "intensity across", "line across", "distance"],
    "Identify an unknown spectrum":
        ["unknown", "identify", "what is", "match", "library", "database"],
    "Measure particle size / count":
        ["particle", "size", "count", "diameter", "grain"],
    "Analyse depth / 3D volume":
        ["3d", "depth", "volume", "z-stack", "confocal", "slice"],
    "Quantify components against reference spectra":
        ["reference", "cls", "dcls", "nnls", "concentration"],
    "PCA / explore variation / classify (PCA Studio)":
        ["pca", "pca studio", "variance", "dimensionality", "score", "loading", "outlier",
         "pls-da", "plsda", "classify", "classification", "hotelling", "t2", "q residual",
         "scree", "snv", "mean centre", "mean center"],
    "Preprocess data / build a reproducible pipeline":
        ["preprocess", "baseline", "cosmic", "smooth", "savitzky", "normalise",
         "normalize", "pipeline", "recipe", "asls"],
    "Check data quality (SNR / intensity / saturation)":
        ["quality", "qc", "snr", "signal to noise", "saturation", "saturated"],
    "Batch-process a folder of maps":
        ["batch", "folder", "many files", "headless", "automate", "pipeline run"],
    "Compare or combine several maps (multi-map)":
        ["multi-map", "multi map", "multiple maps", "several maps", "combine maps"],
    "Edit a map (resample / crop / rotate / subtract background)":
        ["resample", "crop", "rotate", "background subtract", "substrate", "trim"],
    "Extract endmembers / novel regions (N-FINDR, FlyHash)":
        ["endmember", "n-findr", "nfindr", "flyhash", "anomaly", "novel", "purest"],
    "Save a report or session":
        ["report", "html", "session", "save state", "export report"],
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
    try:
        _st = ttk.Style(win)
        for _b in ("TEntry", "TCombobox"):
            _st.configure(_b, foreground="black", fieldbackground="white",
                          background="white", insertcolor="black")
        _st.map("TEntry", foreground=[("!disabled", "black")], fieldbackground=[("!disabled", "white")])
        _st.map("TCombobox", foreground=[("readonly", "black"), ("!disabled", "black")],
                fieldbackground=[("readonly", "white"), ("!disabled", "white")])
    except Exception:
        pass

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
    free_entry = ttk.Entry(r3, textvariable=free, width=54)
    free_entry.pack(side="left")
    free_entry.bind("<Button-1>", lambda e: free_entry.focus_set())
    ttk.Label(top, text="(Free text is matched to the closest goal — offline, no search.)",
              foreground="#666").pack(anchor="w")
    ttk.Label(top, text=f"Still unsure? Use ‘Ask the developer’ below — {DEV_EMAIL} "
                        "(response within ~48 h).", foreground="#2E4468").pack(anchor="w", pady=(4, 0))

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

    def ask_developer():
        import webbrowser, urllib.parse
        subject = "BioRaman — analysis help"
        body = (
            "Hi Akalabya,\n\n"
            "I'm using BioRaman and I'm not sure how to set up my analysis.\n\n"
            f"Sample type: {sample.get()}\n"
            f"Goal: {goal.get()}\n"
            f"Notes: {free.get()}\n\n"
            "My question:\n\n\n"
            "(Attached: data description / screenshots if relevant.)"
        )
        url = "mailto:%s?subject=%s&body=%s" % (
            DEV_EMAIL,
            urllib.parse.quote(subject),
            urllib.parse.quote(body),
        )
        try:
            webbrowser.open(url)
        except Exception:
            pass
        messagebox.showinfo(
            "Ask the developer",
            "Opening your email client to contact the developer:\n\n"
            f"{DEV_EMAIL}\n\nTypical response time: within 48 hours.\n\n"
            "If your email app didn't open, please email the address above directly.")

    bar = ttk.Frame(win, padding=(12, 0, 12, 12)); bar.pack(fill="x")
    ttk.Button(bar, text="🧭 Generate plan", command=generate).pack(side="left")
    ttk.Button(bar, text="⬇ Save plan…", command=save).pack(side="left", padx=6)
    ttk.Button(bar, text="✉ Still not sure? Ask the developer",
               command=ask_developer).pack(side="left", padx=6)
    startup_var = tk.BooleanVar(value=should_show_startup())
    def _toggle():
        d = _load_cfg(); d["show_startup"] = startup_var.get(); _save_cfg(d)
    ttk.Checkbutton(bar, text="Show at startup", variable=startup_var,
                    command=_toggle).pack(side="right")
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right", padx=6)
    generate()
    # macOS: a Toplevel opened at startup may not grab keyboard focus — force it
    def _grab_focus():
        try:
            win.lift(); win.focus_force(); free_entry.focus_set()
        except Exception:
            pass
    win.after(150, _grab_focus)
    return win


if __name__ == "__main__":
    open_planner()
