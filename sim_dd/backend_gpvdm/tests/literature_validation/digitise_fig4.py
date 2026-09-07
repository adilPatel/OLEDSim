#!/usr/bin/env python3
# digitise_fig4.py - extract the four J-V curves from knapp_fig4.png
# (Knapp et al., J. Appl. Phys. 108, 054504 (2010), Fig. 4).
#
# Calibration.  The axis frame sits at cols 274/1687 (V = 1e-2 .. 1e0).  The
# y decades come from the major TICK MARKS, found by scanning inward from the
# left frame for full-length dark runs: rows 204 / 593 / 982, spaced exactly
# 389 / 389 px for 5 decades each -> 77.80 px/decade with 1e0 at row 204.
#
# NB: calibrating instead from the tick LABEL centroids gives row 196.5, which
# is 7.5px (0.096 decades, a factor 1.25) too high - the exponent superscript
# in "10^-5" pulls the centroid up.  The tick marks are the correct anchor.
#
# Extraction.  Each curve is isolated by RGB distance, then TRACKED column by
# column from the right edge: at each column the run closest to the previous
# row is taken.  A plain "topmost" or "longest run" rule fails here because
# neighbouring curves' open marker rims fall inside the colour tolerance and
# capture the trace.  Each curve is therefore seeded with its measured row at
# col 1670.
#
# Two traps that cost real debugging time, recorded so they are not repeated:
#  * The black curve needs tol=90; at tol=60 it loses half its columns.
#  * The legend exclusion box must cover ONLY the swatch lines (cols ~340-505).
#    The red data curve passes horizontally through the legend's row band at
#    low bias, so a full-width box silently truncates it at V=0.088.
#
# Verified by re-projecting the CSVs back onto the source PNG; all four traces
# land on their curves across the full range.
#
# Usage:  python3 digitise_fig4.py     (writes knapp_fig4_d*_digitised.csv)

from PIL import Image
import numpy as np, csv

a = np.array(Image.open('knapp_fig4.png').convert('RGB')).astype(int)

X0, X1 = 274.0, 1687.0
Y0, YV = 204.0, 77.80           # 1e0 row, px/decade (from tick MARKS)
XL, XR = 278, 1684
YT, YB = 55, 1132

def col2V(c): return 10 ** (-2.0 + (c - X0) / ((X1 - X0) / 2.0))
def row2J(r): return 10 ** (0.0 - (r - Y0) / YV)

# tol tightened per colour; the two greys are close so keep them strict
TARGETS = {
    # rgb, tolerance, seed row at col 1670 (measured from the figure)
    "0.00": ((56, 19, 118), 90, 180),
    "0.33": ((26, 25, 25), 90, 223),
    "0.67": ((143, 143, 143), 60, 373),
    "1.00": ((204, 43, 47), 90, 768),
}
# legend swatch lines + markers only (cols 358-500); the red DATA curve
# passes through this row band at low bias, so the box must not span it
LEG = (slice(700, 1132), slice(340, 505))

for name, (rgb, tol, seed) in TARGETS.items():
    m = np.abs(a - np.array(rgb)).sum(2) < tol
    m[:YT, :] = False; m[YB:, :] = False
    m[:, :XL] = False; m[:, XR:] = False
    m[LEG] = False

    # Track the curve continuously: start from the right edge (curves are
    # well separated there) and at each step take the run closest to the
    # previous row, so marker rims of a neighbouring curve can't capture it.
    cols = list(range(XR - 1, XL - 1, -1))
    prev, V, J = None, [], []
    for c in cols:
        rr = np.where(m[:, c])[0]
        if rr.size == 0:
            continue
        runs, cur = [], [rr[0]]
        for v in rr[1:]:
            if v - cur[-1] <= 3: cur.append(v)
            else: runs.append(cur); cur = [v]
        runs.append(cur)
        cand = [np.mean(r) for r in runs]
        if prev is None:
            r = min(cand, key=lambda x: abs(x - seed))   # seeded start
        else:
            r = min(cand, key=lambda x: abs(x - prev))
            if abs(r - prev) > 25:              # tracking lost - skip column
                continue
        prev = r
        V.append(col2V(c)); J.append(row2J(r))

    V, J = np.array(V)[::-1], np.array(J)[::-1]
    if len(J) > 11:
        k = 11
        lg = np.log10(J)
        sm = np.convolve(lg, np.ones(k)/k, mode='same')
        sm[:k] = lg[:k]; sm[-k:] = lg[-k:]
        J = 10**sm
    out = 'knapp_fig4_d%s_digitised.csv' % name.replace('.', 'p')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['V', 'J'])
        for v, j in zip(V, J): w.writerow(['%.6g' % v, '%.6g' % j])
    print('%s: %4d pts  J(0.01)=%.3e  J(0.1)=%.3e  J(1.0)=%.3e'
          % (out, len(V), J[0], J[np.argmin(abs(V-0.1))], J[-1]))
