#!/usr/bin/env python3
"""
Reformat raw SETFOS exports into the validation layout defined in
tests/diode_1d/format.md, so they can be diffed against a DEVSIM simulation's
"<device>_devsim_profiles.txt" / "<device>_devsim_figures.txt" output.

Usage:
    python3 reformat_reference.py <device_name> \
        --profiles <src_raw_profiles.txt> --figures <src_raw_figures.txt> \
        --band-gap EV --vbi VOLTS [--out-dir DIR]

Example:
    python3 reformat_reference.py oled1 \
        --profiles reference_data_raw/setfos_profiles_oled1.txt \
        --figures reference_data_raw/setfos_figures_oled1.txt \
        --band-gap 2.6 --vbi 2.5
"""

import argparse
import os

# Columns in a raw SETFOS profile export (tab-separated, after the leading
# '#'-comment header lines):
# 0 Voltage(V) | 1 x(nm) | 2 Recombination | 3 n(cm^-3) | 4 p(cm^-3)
# | 5 E-field | 6 Potential(V) | 7 J | 8 Jn | 9 Jp | 10 Jdisp
# | 11 MobN(cm^2/Vs) | 12 MobP(cm^2/Vs) | 13 LUMO | 14 HOMO | 15 EFn | 16 EFp
PROFILE_COL_V, PROFILE_COL_X, PROFILE_COL_POTENTIAL, PROFILE_COL_N, PROFILE_COL_P, \
    PROFILE_COL_MUN, PROFILE_COL_MUP = 0, 1, 6, 3, 4, 11, 12

PROFILE_HEADER_ROW = "x (nm) | v (V) | potential (V) | n (cm^-3) | p (cm^-3) | muN (cm^2/Vs) | muP (cm^2/Vs)"

# Columns in a raw SETFOS figures export (tab-separated, after the leading
# '#'-comment header lines):
# 0 Voltage(V) | 1 Built-in voltage(V) | 2 Jmean(mA/cm^2) | ...
FIGURE_COL_V, FIGURE_COL_J = 0, 2

FIGURE_HEADER_ROW = "V-Vbi (V) | J (mA/cm^2)"


def read_data_rows(src_path):
    rows = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


def reformat_profiles(src_path, device_name, band_gap, vbi, dst_path):
    rows = read_data_rows(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("{0}\n".format(device_name))
        f.write("{0}\n".format(band_gap))
        f.write("{0}\n".format(vbi))
        f.write("\n")
        f.write("{0}\n".format(PROFILE_HEADER_ROW))
        prev_v = None
        for fields in rows:
            v = fields[PROFILE_COL_V]
            if prev_v is not None and v != prev_v:
                f.write("\n")
            prev_v = v
            f.write("\t".join([
                fields[PROFILE_COL_X],
                fields[PROFILE_COL_V],
                fields[PROFILE_COL_POTENTIAL],
                fields[PROFILE_COL_N],
                fields[PROFILE_COL_P],
                fields[PROFILE_COL_MUN],
                fields[PROFILE_COL_MUP],
            ]) + "\n")


def reformat_figures(src_path, device_name, vbi, dst_path):
    rows = read_data_rows(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("{0}\n".format(device_name))
        f.write("{0}\n".format(vbi))
        f.write("\n")
        f.write("{0}\n".format(FIGURE_HEADER_ROW))
        for fields in rows:
            v_minus_vbi = float(fields[FIGURE_COL_V]) - vbi
            f.write("{0:e}\t{1}\n".format(v_minus_vbi, fields[FIGURE_COL_J]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device_name", help="device name, used as the header line and default output folder")
    parser.add_argument("--profiles", required=True, help="path to the raw SETFOS profile export")
    parser.add_argument("--figures", required=True, help="path to the raw SETFOS figure export")
    parser.add_argument("--band-gap", type=float, required=True, help="band gap at V=0V (eV)")
    parser.add_argument("--vbi", type=float, required=True, help="built-in voltage at V=0V (V)")
    parser.add_argument("--out-dir", default=None,
                         help="output directory (default: <device_name>/, relative to this script's directory)")
    args = parser.parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, args.device_name)

    profiles_dst = os.path.join(out_dir, "{0}_reference_profiles.txt".format(args.device_name))
    figures_dst = os.path.join(out_dir, "{0}_reference_figures.txt".format(args.device_name))

    reformat_profiles(args.profiles, args.device_name, args.band_gap, args.vbi, profiles_dst)
    reformat_figures(args.figures, args.device_name, args.vbi, figures_dst)

    print("wrote", profiles_dst)
    print("wrote", figures_dst)


if __name__ == "__main__":
    main()
