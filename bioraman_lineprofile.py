"""
BioRaman — Line Profile (line scan) tool
=========================================
Draw a line across a Raman map and plot band intensity (or A/B ratio) versus
distance along the line — the "line scan" used to show, e.g., how a component's
signal changes from the extracellular space into a cell/nucleus
(cf. Louwagie et al., Cryobiology 2026, Fig. 3H–J).

Features
  • one or two bands on a dual axis (Band 1 = left/black, Band 2 = right/red)
  • distance calibrated in µm when the map's pixel size is known, else in pixels
  • optional shading of the region inside the current ROI (intra- vs extra-cellular)
  • export the profile as CSV and the figure as PNG/PDF

Hook into bioraman.py:
    import bioraman_lineprofile as blp
    am.add_command(label="＿ Line Profile (line scan)…",
                   command=lambda: blp.open_line_profile(self))

Author: Akalabya Bissoyi  •  github.com/Akalabyabissoyi/BioRaman
"""
from __future__ import annotations
import os
import json
import numpy as np

# Optional, git-ignored personal presets (kept out of the public build).
_PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bioraman_presets.json")

def _load_lp_presets():
    """Return {name: [b1lo,b1hi,b2lo,b2hi]} from the local presets file, or {}."""
    try:
        return json.load(open(_PRESET_FILE)).get("lineprofile_presets", {})
    except Exception:
        return {}


def _force_black_entries(win):
    """Ensure entry/combobox/spinbox text is dark on white (the app's dark theme
    otherwise renders typed text in a light colour that is hard to read)."""
    try:
        from tkinter import ttk
        st = ttk.Style(win)
        for base in ("TEntry", "TCombobox", "TSpinbox"):
            st.configure(base, foreground="black", fieldbackground="white",
                         background="white", insertcolor="black")
        st.map("TEntry", foreground=[("!disabled", "black")],
               fieldbackground=[("!disabled", "white")])
        st.map("TCombobox",
               foreground=[("readonly", "black"), ("!disabled", "black")],
               fieldbackground=[("readonly", "white"), ("!disabled", "white")])
        st.map("TSpinbox", foreground=[("!disabled", "black")],
               fieldbackground=[("!disabled", "white")])
    except Exception:
        pass


def _band_map(cube, waves, lo, hi):
    """Integrated intensity of a band window for every pixel -> (Y, X)."""
    waves = np.asarray(waves, float)
    m = (waves >= lo) & (waves <= hi)
    if not m.any():
        return np.zeros(cube.shape[:2])
    seg = np.clip(cube[:, :, m], 0, None)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapz(seg, waves[m], axis=2)


def _sample_line(img, x0, y0, x1, y1, n):
    """Sample a 2-D image along a line from (x0,y0) to (x1,y1) at n points."""
    rows = np.linspace(y0, y1, n)
    cols = np.linspace(x0, x1, n)
    try:
        from scipy.ndimage import map_coordinates
        return map_coordinates(img, np.vstack([rows, cols]), order=1, mode="nearest")
    except Exception:
        ri = np.clip(np.round(rows).astype(int), 0, img.shape[0] - 1)
        ci = np.clip(np.round(cols).astype(int), 0, img.shape[1] - 1)
        return img[ri, ci]


def open_line_profile(app):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    cube = getattr(app, "spectra", None)
    waves = getattr(app, "xdata", None)
    if cube is None or waves is None or np.asarray(cube).ndim != 3:
        messagebox.showwarning("No map", "Load a Raman map first (a 2-D map, not a single spectrum).")
        return
    cube = np.asarray(cube, float)
    Y, X, _ = cube.shape
    px_um = getattr(app, "_px_um", None)
    unit = "µm" if px_um else "pixels"
    roi = getattr(app, "_roi_mask", None)
    if roi is not None and np.asarray(roi).shape != (Y, X):
        roi = None

    win = tk.Toplevel(app)
    win.title("BioRaman — Line Profile (line scan)")
    win.geometry("1080x640")
    _force_black_entries(win)
    state = {"pts": [], "fig_prof": None}

    # ── controls (left) ────────────────────────────────────────────────────
    left = ttk.Frame(win, padding=8); left.pack(side="left", fill="y")
    ttk.Label(left, text="Line Profile", font=("", 12, "bold")).pack(anchor="w")
    ttk.Label(left, text="Click two points on the map to set the line.",
              foreground="#555").pack(anchor="w", pady=(0, 6))

    b1 = {"lo": tk.DoubleVar(value=float(np.percentile(waves, 40))),
          "hi": tk.DoubleVar(value=float(np.percentile(waves, 45)))}

    # Band presets are NOT shipped in the public build (BioRaman is general-purpose).
    # They load only from an optional, git-ignored personal file next to this module:
    #   .bioraman_presets.json  ->  {"lineprofile_presets": {"<name>": [b1lo,b1hi,b2lo,b2hi]}}
    LP_PRESETS = {"Custom (enter values)": None}
    for _nm, _v in (_load_lp_presets() or {}).items():
        try:
            LP_PRESETS[_nm] = tuple(float(x) for x in _v)
        except Exception:
            pass
    pf = ttk.Frame(left); pf.pack(fill="x", pady=(2, 2))
    ttk.Label(pf, text="Preset", width=6).pack(side="left")
    preset_var = tk.StringVar(value="Custom (enter values)")
    ttk.Combobox(pf, textvariable=preset_var, values=list(LP_PRESETS),
                 state="readonly", width=34).pack(side="left")

    f1 = ttk.LabelFrame(left, text="Band 1  (black, left axis)", padding=6); f1.pack(fill="x", pady=4)
    for t, k in [("lo", "lo"), ("hi", "hi")]:
        r = ttk.Frame(f1); r.pack(fill="x")
        ttk.Label(r, text=t, width=4).pack(side="left")
        ttk.Entry(r, textvariable=b1[k], width=9).pack(side="left")

    use2 = tk.BooleanVar(value=True)   # dual-axis by default (Band 1 black / Band 2 red)
    ttk.Checkbutton(left, text="Add Band 2 (red, right axis)", variable=use2).pack(anchor="w", pady=(6, 0))
    b2 = {"lo": tk.DoubleVar(value=float(np.percentile(waves, 70))),
          "hi": tk.DoubleVar(value=float(np.percentile(waves, 75)))}
    f2 = ttk.LabelFrame(left, text="Band 2  (red, right axis)", padding=6); f2.pack(fill="x", pady=4)
    for t, k in [("lo", "lo"), ("hi", "hi")]:
        r = ttk.Frame(f2); r.pack(fill="x")
        ttk.Label(r, text=t, width=4).pack(side="left")
        ttk.Entry(r, textvariable=b2[k], width=9).pack(side="left")

    def apply_preset(*_):
        v = LP_PRESETS.get(preset_var.get())
        if not v:
            return
        b1["lo"].set(v[0]); b1["hi"].set(v[1])
        b2["lo"].set(v[2]); b2["hi"].set(v[3])
        use2.set(True)
    preset_var.trace_add("write", apply_preset)

    shade_var = tk.BooleanVar(value=(roi is not None))
    cb_shade = ttk.Checkbutton(left, text="Shade inside ROI (intra-cellular)", variable=shade_var)
    cb_shade.pack(anchor="w", pady=(4, 0))
    if roi is None:
        cb_shade.state(["disabled"])
        ttk.Label(left, text="(draw an ROI in the main window to enable shading)",
                  foreground="#888").pack(anchor="w")

    ttk.Label(left, text=f"Pixel size: {('%.3f µm' % px_um) if px_um else 'unknown → axis in pixels'}",
              foreground="#555").pack(anchor="w", pady=(8, 0))

    # ── figures (right): map (top) + profile (bottom) ──────────────────────
    right = ttk.Frame(win); right.pack(side="right", fill="both", expand=True)
    fig_map = Figure(figsize=(4.6, 3.0)); axm = fig_map.add_subplot(111)
    dmap = _band_map(cube, waves, b1["lo"].get(), b1["hi"].get())
    im = axm.imshow(dmap, origin="upper", aspect="equal", cmap="turbo", interpolation="bicubic")
    axm.set_title("Band-1 map — click 2 points", fontsize=9)
    axm.set_xlabel("Pixel in x", fontsize=8); axm.set_ylabel("Pixel in y", fontsize=8)
    cvm = FigureCanvasTkAgg(fig_map, master=right); cvm.draw()
    cvm.get_tk_widget().pack(fill="both", expand=True)

    fig_prof = Figure(figsize=(4.6, 3.0)); axp = fig_prof.add_subplot(111)
    cvp = FigureCanvasTkAgg(fig_prof, master=right); cvp.draw()
    cvp.get_tk_widget().pack(fill="both", expand=True)
    state["fig_prof"] = fig_prof

    line_artist = {"ln": None, "pts": []}

    def refresh_map(*_):
        d = _band_map(cube, waves, b1["lo"].get(), b1["hi"].get())
        im.set_data(d); im.set_clim(np.nanmin(d), np.nanmax(d)); cvm.draw_idle()
    for v in (b1["lo"], b1["hi"]):
        v.trace_add("write", refresh_map)

    def on_click(ev):
        if ev.inaxes is not axm or ev.xdata is None:
            return
        if len(state["pts"]) >= 2:
            state["pts"] = []
            for a in line_artist["pts"]:
                a.remove()
            line_artist["pts"] = []
            if line_artist["ln"] is not None:
                line_artist["ln"].remove(); line_artist["ln"] = None
        state["pts"].append((ev.xdata, ev.ydata))
        p, = axm.plot(ev.xdata, ev.ydata, "wo", ms=5, mec="k")
        line_artist["pts"].append(p)
        if len(state["pts"]) == 2:
            (x0, y0), (x1, y1) = state["pts"]
            line_artist["ln"], = axm.plot([x0, x1], [y0, y1], "w-", lw=1.2)
        cvm.draw_idle()
    fig_map.canvas.mpl_connect("button_press_event", on_click)

    def plot_profile():
        if len(state["pts"]) != 2:
            messagebox.showinfo("Set a line", "Click two points on the map first."); return
        (x0, y0), (x1, y1) = state["pts"]
        n = max(20, int(round(np.hypot(x1 - x0, y1 - y0))) + 1)
        step_px = np.hypot(x1 - x0, y1 - y0) / (n - 1)
        dist = np.arange(n) * step_px * (px_um if px_um else 1.0)
        m1 = _band_map(cube, waves, b1["lo"].get(), b1["hi"].get())
        p1 = _sample_line(m1, x0, y0, x1, y1, n)
        axp.clear()
        axp.plot(dist, p1, color="k", lw=1.6, label="Band 1")
        axp.set_xlabel(f"Distance along line ({unit})", fontsize=9)
        axp.set_ylabel("Band 1 intensity (a.u.)", fontsize=9)
        cols = {"dist": dist, "band1": p1}
        ax2 = None
        if use2.get():
            m2 = _band_map(cube, waves, b2["lo"].get(), b2["hi"].get())
            p2 = _sample_line(m2, x0, y0, x1, y1, n)
            ax2 = axp.twinx()
            ax2.plot(dist, p2, color="#d1495b", lw=1.6, label="Band 2")
            ax2.set_ylabel("Band 2 intensity (a.u.)", color="#d1495b", fontsize=9)
            ax2.tick_params(axis="y", colors="#d1495b")
            cols["band2"] = p2
        if shade_var.get() and roi is not None:
            inside = _sample_line(roi.astype(float), x0, y0, x1, y1, n) >= 0.5
            # shade contiguous inside-ROI segments
            d = np.concatenate([[0], np.diff(inside.astype(int))])
            starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
            if inside[0]:
                starts = np.r_[0, starts]
            if inside[-1]:
                ends = np.r_[ends, n - 1]
            for s, e in zip(starts, ends):
                axp.axvspan(dist[s], dist[min(e, n - 1)], color="0.85", zorder=0)
            cols["inside_roi"] = inside.astype(int)
        axp.set_title("Line scan", fontsize=9)
        fig_prof.tight_layout(); cvp.draw()
        state["profile"] = cols

    def export():
        if "profile" not in state:
            messagebox.showinfo("Nothing yet", "Plot a profile first."); return
        d = filedialog.askdirectory(title="Export line profile to…")
        if not d:
            return
        cols = state["profile"]
        keys = list(cols)
        with open(os.path.join(d, "line_profile.csv"), "w") as fh:
            fh.write(",".join(keys) + "\n")
            for i in range(len(cols["dist"])):
                fh.write(",".join(f"{cols[k][i]:.6g}" for k in keys) + "\n")
        state["fig_prof"].savefig(os.path.join(d, "line_profile.png"), dpi=400, bbox_inches="tight")
        state["fig_prof"].savefig(os.path.join(d, "line_profile.pdf"), bbox_inches="tight")
        messagebox.showinfo("Exported", "Saved line_profile.csv / .png / .pdf to\n" + d)

    ttk.Button(left, text="▶ Plot profile", command=plot_profile).pack(fill="x", pady=(12, 2))
    ttk.Button(left, text="⬇ Export CSV + figure", command=export).pack(fill="x")
    ttk.Button(left, text="Reset line", command=lambda: (state.__setitem__("pts", []), refresh_map())).pack(fill="x", pady=(6, 0))
    return win
