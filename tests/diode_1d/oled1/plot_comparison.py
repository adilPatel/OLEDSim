#!/usr/bin/env python3
"""
Interactive comparison of DEVSIM vs SETFOS reference profiles for the oled1
test case, read from the format.md-layout files in this folder:
    oled1_devsim_profiles.txt
    oled1_reference_profiles.txt

Opens four figures, each with a voltage slider:
    1. Electron/hole density vs x
    2. Potential vs x
    3. Electric field vs x (computed as -dPotential/dx)
    4. Current density (|J|) vs V-Vbi, semilogy

Usage:
    python3 plot_comparison.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVSIM_PATH = os.path.join(SCRIPT_DIR, "oled1_devsim_profiles.txt")
REFERENCE_PATH = os.path.join(SCRIPT_DIR, "oled1_reference_profiles.txt")
DEVSIM_FIGURES_PATH = os.path.join(SCRIPT_DIR, "oled1_devsim_figures.txt")
REFERENCE_FIGURES_PATH = os.path.join(SCRIPT_DIR, "oled1_reference_figures.txt")


def load_profiles(path):
    """
    Parses a format.md-layout profile file into a dict keyed by voltage,
    each value a dict of column -> list of floats, sorted by x.
    """
    with open(path) as f:
        lines = f.readlines()

    device_name = lines[0].strip()
    data_lines = lines[5:]

    blocks = []
    current = []
    for line in data_lines:
        line = line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line.split("\t"))
    if current:
        blocks.append(current)

    profiles = {}
    for block in blocks:
        v = round(float(block[0][1]), 6)
        rows = sorted(block, key=lambda r: float(r[0]))
        profiles[v] = {
            "x": [float(r[0]) for r in rows],
            "potential": [float(r[2]) for r in rows],
            "n": [float(r[3]) for r in rows],
            "p": [float(r[4]) for r in rows],
        }
    return device_name, profiles


def load_figures(path):
    """
    Parses a format.md-layout figure file (device name, Vbi, blank line,
    header, then "V-Vbi (V) | J (mA/cm^2)" rows) into sorted lists of
    (v_minus_vbi, J).
    """
    with open(path) as f:
        lines = f.readlines()

    device_name = lines[0].strip()
    vbi = float(lines[1].strip())
    rows = []
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        v_minus_vbi, j = line.split("\t")
        rows.append((float(v_minus_vbi), float(j)))
    rows.sort(key=lambda r: r[0])
    return device_name, vbi, rows


def main():
    devsim_name, devsim_profiles = load_profiles(DEVSIM_PATH)
    reference_name, reference_profiles = load_profiles(REFERENCE_PATH)
    _, devsim_vbi, devsim_iv = load_figures(DEVSIM_FIGURES_PATH)
    _, _, reference_iv = load_figures(REFERENCE_FIGURES_PATH)

    voltages = sorted(set(devsim_profiles) & set(reference_profiles))
    if not voltages:
        raise RuntimeError("No common voltages found between DEVSIM and reference profiles")

    def efield_kv_per_cm(profile):
        """
        E = -dPotential/dx, via centered (interior) / one-sided (endpoint)
        finite differences on the profile's own x grid. x is in nm and
        Potential in V, so dPotential/dx is in V/nm; convert to kV/cm
        (1 V/nm = 1e7 V/cm = 1e4 kV/cm) to match the field magnitudes
        typically reported for these devices.
        """
        x = profile["x"]
        phi = profile["potential"]
        n = len(x)
        field = [0.0] * n
        nm_to_cm = 1e-7
        for i in range(n):
            if i == 0:
                dphi = phi[1] - phi[0]
                dx = x[1] - x[0]
            elif i == n - 1:
                dphi = phi[-1] - phi[-2]
                dx = x[-1] - x[-2]
            else:
                dphi = phi[i + 1] - phi[i - 1]
                dx = x[i + 1] - x[i - 1]
            field[i] = -(dphi / dx) / nm_to_cm / 1000.0
        return field

    ####
    #### Figure 1: Electron/hole density vs x
    ####
    fig1, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    line_devsim_n, = ax1.semilogy(devsim_profiles[voltages[0]]["x"], devsim_profiles[voltages[0]]["n"],
                                   label="{0} Electrons".format(devsim_name), color="tab:blue", linestyle="-")
    line_devsim_p, = ax1.semilogy(devsim_profiles[voltages[0]]["x"], devsim_profiles[voltages[0]]["p"],
                                   label="{0} Holes".format(devsim_name), color="tab:red", linestyle="-")
    line_ref_n, = ax1.semilogy(reference_profiles[voltages[0]]["x"], reference_profiles[voltages[0]]["n"],
                                label="{0} Electrons".format(reference_name), color="tab:blue", linestyle="--",
                                marker="o", markersize=4)
    line_ref_p, = ax1.semilogy(reference_profiles[voltages[0]]["x"], reference_profiles[voltages[0]]["p"],
                                label="{0} Holes".format(reference_name), color="tab:red", linestyle="--",
                                marker="o", markersize=4)
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("Density (cm^-3)")
    ax1.legend()
    ax1.set_title("V = {0:.2f} V".format(voltages[0]))

    all_densities = []
    for profile in list(devsim_profiles.values()) + list(reference_profiles.values()):
        all_densities.extend(v for v in profile["n"] + profile["p"] if v > 0)
    ax1.set_ylim(min(all_densities), max(all_densities) * 10)
    ax1.set_xlim(0, max(devsim_profiles[voltages[0]]["x"]))

    slider1_ax = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider1 = Slider(slider1_ax, "Voltage index", 0, len(voltages) - 1, valinit=0, valstep=1)

    def update1(val):
        idx = int(slider1.val)
        v = voltages[idx]
        line_devsim_n.set_data(devsim_profiles[v]["x"], devsim_profiles[v]["n"])
        line_devsim_p.set_data(devsim_profiles[v]["x"], devsim_profiles[v]["p"])
        line_ref_n.set_data(reference_profiles[v]["x"], reference_profiles[v]["n"])
        line_ref_p.set_data(reference_profiles[v]["x"], reference_profiles[v]["p"])
        ax1.set_title("V = {0:.2f} V".format(v))
        fig1.canvas.draw_idle()

    slider1.on_changed(update1)

    ####
    #### Figure 2: Potential vs x
    ####
    fig2, ax2 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    line_devsim_pot, = ax2.plot(devsim_profiles[voltages[0]]["x"], devsim_profiles[voltages[0]]["potential"],
                                 label=devsim_name, color="tab:purple", linestyle="-")
    line_ref_pot, = ax2.plot(reference_profiles[voltages[0]]["x"], reference_profiles[voltages[0]]["potential"],
                              label=reference_name, color="tab:brown", linestyle="--",
                              marker="o", markersize=4)
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("Potential (V)")
    ax2.legend()
    ax2.set_title("V = {0:.2f} V".format(voltages[0]))

    all_potentials = []
    for profile in list(devsim_profiles.values()) + list(reference_profiles.values()):
        all_potentials.extend(profile["potential"])
    ax2.set_ylim(min(all_potentials), max(all_potentials))
    ax2.set_xlim(0, max(devsim_profiles[voltages[0]]["x"]))

    slider2_ax = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider2 = Slider(slider2_ax, "Voltage index", 0, len(voltages) - 1, valinit=0, valstep=1)

    def update2(val):
        idx = int(slider2.val)
        v = voltages[idx]
        line_devsim_pot.set_data(devsim_profiles[v]["x"], devsim_profiles[v]["potential"])
        line_ref_pot.set_data(reference_profiles[v]["x"], reference_profiles[v]["potential"])
        ax2.set_title("V = {0:.2f} V".format(v))
        fig2.canvas.draw_idle()

    slider2.on_changed(update2)

    ####
    #### Figure 3: Electric field vs x
    ####
    fig3, ax3 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    line_devsim_field, = ax3.plot(devsim_profiles[voltages[0]]["x"], efield_kv_per_cm(devsim_profiles[voltages[0]]),
                                   label=devsim_name, color="tab:purple", linestyle="-")
    line_ref_field, = ax3.plot(reference_profiles[voltages[0]]["x"], efield_kv_per_cm(reference_profiles[voltages[0]]),
                                label=reference_name, color="tab:brown", linestyle="--",
                                marker="o", markersize=4)
    ax3.set_xlabel("x (nm)")
    ax3.set_ylabel("E-field (kV/cm)")
    ax3.legend()
    ax3.set_title("V = {0:.2f} V".format(voltages[0]))

    all_fields = []
    for profile in list(devsim_profiles.values()) + list(reference_profiles.values()):
        all_fields.extend(efield_kv_per_cm(profile))
    ax3.set_ylim(min(all_fields), max(all_fields))
    ax3.set_xlim(0, max(devsim_profiles[voltages[0]]["x"]))

    slider3_ax = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider3 = Slider(slider3_ax, "Voltage index", 0, len(voltages) - 1, valinit=0, valstep=1)

    def update3(val):
        idx = int(slider3.val)
        v = voltages[idx]
        line_devsim_field.set_data(devsim_profiles[v]["x"], efield_kv_per_cm(devsim_profiles[v]))
        line_ref_field.set_data(reference_profiles[v]["x"], efield_kv_per_cm(reference_profiles[v]))
        ax3.set_title("V = {0:.2f} V".format(v))
        fig3.canvas.draw_idle()

    slider3.on_changed(update3)

    ####
    #### Figure 4: Current density (|J|) vs V-Vbi, semilogy
    ####
    fig4, ax4 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    devsim_v_minus_vbi = [row[0] for row in devsim_iv]
    devsim_abs_j = [abs(row[1]) for row in devsim_iv]
    reference_v_minus_vbi = [row[0] for row in reference_iv]
    reference_abs_j = [abs(row[1]) for row in reference_iv]

    ax4.semilogy(devsim_v_minus_vbi, devsim_abs_j, label=devsim_name, color="tab:purple", linestyle="-")
    ax4.semilogy(reference_v_minus_vbi, reference_abs_j, label=reference_name, color="tab:brown", linestyle="--",
                 marker="o", markersize=4)
    marker_line4 = ax4.axvline(voltages[0] - devsim_vbi, color="gray", linewidth=1, linestyle=":")
    ax4.set_xlabel("V - Vbi (V)")
    ax4.set_ylabel("|J| (mA/cm^2)")
    ax4.legend()
    ax4.set_title("V = {0:.2f} V".format(voltages[0]))

    all_abs_j = [j for j in devsim_abs_j + reference_abs_j if j > 0]
    ax4.set_ylim(min(all_abs_j), max(all_abs_j) * 10)
    ax4.set_xlim(min(devsim_v_minus_vbi + reference_v_minus_vbi), max(devsim_v_minus_vbi + reference_v_minus_vbi))

    slider4_ax = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider4 = Slider(slider4_ax, "Voltage index", 0, len(voltages) - 1, valinit=0, valstep=1)

    def update4(val):
        idx = int(slider4.val)
        v = voltages[idx]
        marker_line4.set_xdata([v - devsim_vbi, v - devsim_vbi])
        ax4.set_title("V = {0:.2f} V".format(v))
        fig4.canvas.draw_idle()

    slider4.on_changed(update4)

    plt.show()


if __name__ == "__main__":
    main()
