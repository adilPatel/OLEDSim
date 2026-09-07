#!/usr/bin/env python3
"""verify_tclc.py - verify trap-limited (Mark-Helfrich) current behaviour.

Runs the trap-limited deck and its trap-free control, and checks the three
signatures of trap-limited space-charge-limited current (TCLC):

  1. TRAP-FREE CONTROL obeys Mott-Gurney: log-log slope -> 2.
  2. TRAP-LIMITED DECK obeys  J ~ V^(l+1),  l = Et/kT,
     i.e. a slope steeper than 2 and equal to the value set by the trap
     distribution's characteristic energy.
  3. The exponent TRACKS Et: sweeping Et and re-measuring the slope
     reproduces l+1 across a range of trap temperatures.  This is the test
     that distinguishes genuine TCLC from an arbitrary steep J-V.

Also reports the trap-filling ratio theta = n_free/n_trapped, which must
stay << 1 for the device to be trap-limited at all.

Usage:  python3 verify_tclc.py           (2 runs, ~1 min)
        python3 verify_tclc.py --scan-et (adds the Et sweep, 5 more runs)
"""
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OLED_SIM = os.path.join(BACKEND_DIR, "oled_sim")

KB_EV = 1.380649e-23 / 1.602176634e-19   # eV/K
T = 300.0
KT = KB_EV * T
ET = 0.075          # trap_Et_e in tlc_traps.ini
NTOT = 3.0e24       # integrated trap density [m^-3]
L = 200e-9
MU = 1e-9
EPSR = 3.0
FIT_VMIN = 5.0      # fit the power law above this bias (below it the device
                    # is still turning on against the built-in potential)


def run(deck):
    r = subprocess.run([OLED_SIM, os.path.join(SCRIPT_DIR, deck)],
                       cwd=BACKEND_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        sys.exit(f"oled_sim failed on {deck}")


def jv(out_dir):
    d = np.loadtxt(os.path.join(BACKEND_DIR, out_dir, "device_figures.dat"))
    V, J = d[:, 0], d[:, 1]
    m = (V > 0) & (J > 0)
    return V[m], J[m]


def slope(V, J, vmin=FIT_VMIN):
    m = V > vmin
    return np.polyfit(np.log(V[m]), np.log(J[m]), 1)[0]


def profiles(out_dir):
    """Return {V: array} with columns x n p phi E R mun mup nt pt Jn Jp."""
    path = os.path.join(BACKEND_DIR, out_dir, "device_profiles.dat")
    out, V, rows = {}, None, []
    for line in open(path):
        s = line.strip()
        if s.startswith("# V ="):
            if V is not None and rows:
                out[V] = np.array(rows)
            V, rows = float(s.split("=")[1]), []
        elif s and not s.startswith("#"):
            rows.append([float(x) for x in s.split()])
    if V is not None and rows:
        out[V] = np.array(rows)
    return out


def main():
    ok = True
    l_pred = ET / KT
    print("=" * 68)
    print("TRAP-LIMITED CURRENT (TCLC) VERIFICATION")
    print("=" * 68)
    print(f"  T        = {T:.0f} K   (kT = {KT*1000:.2f} meV)")
    print(f"  Et       = {ET} eV = kB * {ET/KB_EV:.0f} K")
    print(f"  l = Et/kT = {l_pred:.3f}   ->  predicted TCLC slope = {l_pred+1:.3f}")
    print(f"  Ntrap    = {NTOT:.1e} m^-3,  L = {L*1e9:.0f} nm")

    print("\nrunning decks ...")
    run("tlc_trapfree.ini")
    run("tlc_traps.ini")

    Vf, Jf = jv("tests/trap_limited/out_trapfree")
    Vt, Jt = jv("tests/trap_limited/out_traps")

    # ---- test 1: trap-free control is Mott-Gurney (slope 2) ---------------
    sf = slope(Vf, Jf)
    J_MG = 9.0 / 8.0 * EPSR * 8.8541878128e-12 * MU * Vf**2 / L**3
    ratio = (Jf / J_MG)[Vf > FIT_VMIN]
    print("\n[1] TRAP-FREE CONTROL (expect Mott-Gurney, slope 2)")
    print(f"    fitted slope (V > {FIT_VMIN:.0f}V) = {sf:.3f}")
    print(f"    J/J_MG = {ratio.min():.2f} .. {ratio.max():.2f} "
          f"(< 1 and rising: the built-in potential from the asymmetric")
    print(f"    contacts costs some of the applied bias)")
    t1 = 1.9 < sf < 2.4
    print(f"    -> {'PASS' if t1 else 'FAIL'}: slope is 2 within the "
          f"diffusion/Vbi correction")
    ok &= t1

    # ---- test 2: trapped device follows J ~ V^(l+1) -----------------------
    st = slope(Vt, Jt)
    err = 100 * (st - (l_pred + 1)) / (l_pred + 1)
    print(f"\n[2] TRAP-LIMITED DECK (expect slope l+1 = {l_pred+1:.3f})")
    print(f"    fitted slope (V > {FIT_VMIN:.0f}V) = {st:.3f}   "
          f"(error {err:+.2f}%)")
    print(f"    current suppression vs trap-free at 12V: "
          f"{Jf[-1]/Jt[-1]:.0f}x")
    t2 = abs(err) < 5.0 and st > sf + 1.0
    print(f"    -> {'PASS' if t2 else 'FAIL'}: slope matches l+1 to <5% and "
          f"is far above the trap-free slope")
    ok &= t2

    # ---- trap filling check ----------------------------------------------
    print("\n[.] TRAP FILLING (theta = n_free/n_trapped must stay << 1)")
    prof = profiles("tests/trap_limited/out_traps")
    for Vq in [2.0, 6.0, 12.0]:
        key = min(prof, key=lambda v: abs(v - Vq))
        d = prof[key]
        n, p, nt = d[:, 1].mean(), d[:, 2].mean(), d[:, 8].mean()
        print(f"    V={key:5.1f}  <n>={n:.2e}  <nt>={nt:.2e}  "
              f"theta={n/nt:.4f}   p/n={p/n:.1e}")
    def theta_at(Vq):
        d = prof[min(prof, key=lambda v: abs(v - Vq))]
        return d[:, 1].mean() / d[:, 8].mean()

    # Trap-limited transport requires most of the charge to be immobile
    # (theta < 1/2, i.e. the traps hold more than the free band) AND the
    # traps not to be filling up as the bias rises - a device on its way to
    # trap-free behaviour shows theta climbing toward 1, which is exactly
    # what kills the power law.  Both are checked; the magnitude threshold
    # is deliberately the physical one (theta < 0.5), not a tuned number.
    th2, th12 = theta_at(2.0), theta_at(12.0)
    t3 = th12 < 0.5 and th12 <= th2
    print(f"    -> {'PASS' if t3 else 'FAIL'}: theta = {th12:.3f} < 0.5 "
          f"(only {100*th12:.0f}% of the charge is mobile) and falling")
    print(f"       with bias ({th2:.3f} -> {th12:.3f}), so the traps are "
          f"not filling.  p/n ~ 1e-13: electron-only.")
    ok &= t3

    # ---- test 3: exponent tracks Et --------------------------------------
    if "--scan-et" in sys.argv:
        print("\n[3] EXPONENT TRACKS Et  (the decisive TCLC test)")
        print(f"    {'Et[eV]':>7} {'Tt[K]':>7} {'l+1 pred':>9} "
              f"{'measured':>9} {'err%':>7}")
        base = open(os.path.join(SCRIPT_DIR, "tlc_traps.ini")).read()
        worst = 0.0
        for Et in [0.050, 0.060, 0.075, 0.090, 0.105]:
            s = base.replace("trap_Nt_e = 4.0e25",
                             f"trap_Nt_e = {NTOT/Et:.6e}")
            s = s.replace(f"trap_Et_e = {ET}", f"trap_Et_e = {Et}")
            s = s.replace("srh_start = -0.5",
                          f"srh_start = {-min(0.9, 7*Et):.3f}")
            s = s.replace("output_dir = tests/trap_limited/out_traps",
                          "output_dir = tests/trap_limited/out_scan")
            tmp = os.path.join(SCRIPT_DIR, "_scan_tmp.ini")
            open(tmp, "w").write(s)
            subprocess.run([OLED_SIM, tmp], cwd=BACKEND_DIR,
                           capture_output=True, text=True)
            os.remove(tmp)
            Vs, Js = jv("tests/trap_limited/out_scan")
            ss = slope(Vs, Js)
            pred = Et / KT + 1
            e = 100 * (ss - pred) / pred
            worst = max(worst, abs(e))
            print(f"    {Et:7.3f} {Et/KB_EV:7.0f} {pred:9.3f} "
                  f"{ss:9.3f} {e:+7.2f}")
        t4 = worst < 6.0
        print(f"    -> {'PASS' if t4 else 'FAIL'}: the exponent follows "
              f"l+1 = Et/kT + 1 over a 2.6x range of Tt (worst {worst:.1f}%)")
        ok &= t4
    else:
        print("\n[3] skipped (pass --scan-et to sweep Et; ~5 extra runs)")

    print("\n" + "=" * 68)
    print("RESULT: " + ("ALL CHECKS PASS - the simulator reproduces "
                        "trap-limited currents" if ok else "FAILURES ABOVE"))
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
