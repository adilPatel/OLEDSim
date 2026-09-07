#!/usr/bin/env python3
"""Overlay oled_sim carrier densities on the commercial-TCAD reference data.

For every anode work function folder, one figure per mu0n (1e-3, 1e-2,
1e-1 cm2/Vs): free electron and hole densities against position at three
biases, solid lines from this simulator and open markers from the reference.
Also writes an accuracy summary over the whole set.

Needs matplotlib.  The Homebrew python3 here is PEP 668 externally-managed,
so run it from a venv, e.g.
    python3 -m venv --without-pip /tmp/v && ...
or simply:  <venv>/bin/python make_plots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

ANODES = {"4d9": 4.9, "5d1": 5.1, "5d2": 5.2, "5d3": 5.3,
          "5d5": 5.5, "5d6": 5.6, "6d3": 6.3}
OHMIC_ANODES = {"6d3"}
MU = [("0d001", r"$10^{-3}$"), ("0d01", r"$10^{-2}$"), ("0d1", r"$10^{-1}$")]
BIASES = [2.0, 5.0, 8.0]

# validated categorical palette (dataviz six-checks, light surface): lightness
# band / chroma floor / CVD separation / normal-vision floor / contrast all PASS
COLORS = ["#3b5bdb", "#c2410c", "#0ca678"]
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#dcdcdc"


def load_reference(folder, label, carrier):
    """Reference file -> {V: (x_nm, density_cm^-3)}; all-NaN biases dropped."""
    path = os.path.join(HERE, "devices_%s" % folder,
                        "Densities_%s_%s_%s_data.txt" % (folder, label, carrier))
    if not os.path.exists(path):
        return {}
    x, v, d = [], [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        a = line.split()
        if len(a) < 3:
            continue
        x.append(float(a[0])); v.append(float(a[1]))
        try:
            d.append(float(a[2]))
        except ValueError:
            d.append(np.nan)          # the tool emits NaN for failed biases
    x, v, d = np.array(x), np.array(v), np.array(d)
    out = {}
    for bias in BIASES:
        m = np.isclose(v, bias, atol=1e-6)
        if not m.any():
            continue
        xs, ds = x[m], d[m]
        if np.all(np.isnan(ds)):      # whole bias failed to converge
            continue
        o = np.argsort(xs)
        out[bias] = (xs[o], ds[o])
    return out


def load_sim(folder, label):
    """device_profiles -> {V: (x_nm, n_cm^-3, p_cm^-3)}."""
    path = os.path.join(HERE, "devices_%s" % folder,
                        "device_profiles_%s_%s.dat" % (folder, label))
    if not os.path.exists(path):
        return {}
    blocks, cur, V = {}, [], None
    for line in open(path):
        s = line.strip()
        if s.startswith("# V ="):
            if V is not None and cur:
                blocks[V] = np.array(cur)
            V = float(s.split("=")[1]); cur = []
        elif s and not s.startswith("#"):
            cur.append([float(t) for t in s.split()])
    if V is not None and cur:
        blocks[V] = np.array(cur)

    out = {}
    for bias in BIASES:
        key = min(blocks, key=lambda k: abs(k - bias))
        if abs(key - bias) > 0.051:
            continue
        a = blocks[key]
        out[bias] = (a[:, 0] * 1e9, a[:, 1] * 1e-6, a[:, 2] * 1e-6)
    return out


def panel(ax, sim, ref, carrier, title):
    idx = 1 if carrier == "n" else 2
    for c, bias in zip(COLORS, BIASES):
        if bias in sim:
            xs, n, p = sim[bias]
            ax.semilogy(xs, (n if idx == 1 else p), "-", color=c, lw=2.0, zorder=3)
        if bias in ref:
            rx, rd = ref[bias]
            k = max(1, len(rx) // 22)      # thin the markers so lines stay readable
            ax.semilogy(rx[::k], rd[::k], "o", mfc="none", mec=c, mew=1.6,
                        ms=6, ls="none", zorder=4)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel("Position  $x$  (nm)", fontsize=10, color=INK2)
    ax.set_xlim(0, 100)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5, zorder=0)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def figure(folder, label, mutex):
    sim = load_sim(folder, label)
    refn = load_reference(folder, label, "n")
    refp = load_reference(folder, label, "p")
    if not sim:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], sim, refn, "n", "Free electrons  $n(x)$")
    panel(axes[1], sim, refp, "p", "Free holes  $p(x)$")
    axes[0].set_ylabel(r"Carrier density  (cm$^{-3}$)", fontsize=10, color=INK2)

    vals = []
    for d in (refn, refp):
        for _, r in d.items():
            good = r[1][np.isfinite(r[1]) & (r[1] > 0)]
            if good.size:
                vals.append(good)
    for b in sim:
        _, n, p = sim[b]
        vals += [n[n > 0], p[p > 0]]
    allv = np.concatenate(vals)
    hi = 10 ** np.ceil(np.log10(allv.max()))
    # the contact depletion tails run off to ~1e-20 and carry no information;
    # show the nine decades below the peak instead
    axes[0].set_ylim(hi / 1e9, hi * 3)

    handles = [Line2D([], [], color=c, lw=2.0, label="%.0f V" % b)
               for c, b in zip(COLORS, BIASES)]
    handles += [Line2D([], [], color=INK2, lw=2.0, label="This work (oled_sim)"),
                Line2D([], [], color=INK2, marker="o", mfc="none", mew=1.6,
                       ls="none", ms=6, label="Reference TCAD")]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, labelcolor=INK2,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.012))

    amodel = "ohmic" if folder in OHMIC_ANODES else "thermionic"
    fig.suptitle(r"F8BT 100 nm EGDM,  anode %.1f eV (%s),  $\mu_{0n}$ = %s cm$^2$/Vs"
                 % (ANODES[folder], amodel, mutex),
                 fontsize=12.5, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    out = os.path.join(HERE, "devices_%s" % folder,
                       "densities_%s_%s.png" % (folder, label))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def accuracy(folder, label):
    """Mean |log10(sim/ref)| over the device interior, both carriers."""
    sim = load_sim(folder, label)
    errs = []
    for carrier, col in (("n", 1), ("p", 2)):
        ref = load_reference(folder, label, carrier)
        for bias, (rx, rd) in ref.items():
            if bias not in sim:
                continue
            xs = sim[bias][0]
            S = sim[bias][col]
            for xt in range(10, 96, 5):
                j = np.argmin(np.abs(rx - xt))
                r = rd[j]
                if not np.isfinite(r) or r <= 1e12:
                    continue
                i = np.argmin(np.abs(xs - xt))
                if S[i] > 0:
                    errs.append(np.log10(S[i] / r))
    return np.array(errs)


def main():
    made, summary, pooled = 0, [], []
    for folder in sorted(ANODES):
        if not os.path.isdir(os.path.join(HERE, "devices_%s" % folder)):
            continue
        for label, mutex in MU:
            out = figure(folder, label, mutex)
            if out:
                made += 1
                print("wrote", os.path.relpath(out, HERE))
            e = accuracy(folder, label)
            if e.size:
                summary.append((folder, label, e))
                pooled += list(e)

    print("\n%d figures written\n" % made)
    print("Agreement vs reference TCAD  (x = 10-95 nm, densities > 1e12 cm^-3,")
    print("both carriers, 2/5/8 V).  'median dev' is the typical factor error.\n")
    print("%-8s %-8s %12s %14s %8s" % ("anode", "mu0n", "median dev", "mean |log10|", "pts"))
    for folder, label, e in summary:
        med = 10 ** np.median(np.abs(e))
        print("%-8s %-8s %11.0f%% %14.3f %8d"
              % (folder, label, (med - 1) * 100, np.mean(np.abs(e)), e.size))
    pooled = np.array(pooled)
    if pooled.size:
        print("\nOverall: median deviation %.0f%%, mean |log10| %.3f  (%d points)"
              % ((10 ** np.median(np.abs(pooled)) - 1) * 100,
                 np.mean(np.abs(pooled)), pooled.size))


if __name__ == "__main__":
    main()
