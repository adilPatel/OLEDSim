#!/bin/bash
# run_tests.sh - test suite for the backend_gpvdm OLED simulator.
# Run from backend_gpvdm/tests (or via `make test` in backend_gpvdm/).
#
#  1. sclc          hole-only device vs the analytic Mott-Gurney law
#  2. gpvdm_match   JV of gpvdm's default material vs a stored gpvdm v8.0
#                   reference (tests/reference_gpvdm_jv.csv, produced by
#                   make_gpvdm_ref.py with the real gpvdm_core binary)
#  3. oled          3-layer OLED: convergence, recombination confinement,
#                   outcoupling output
#
# To regenerate the gpvdm reference from a built gpvdm_core:
#   python3 make_gpvdm_ref.py gpvdm_ref && cp gpvdm_ref/jv_internal.csv reference_gpvdm_jv.csv

set -e
cd "$(dirname "$0")"
SIM=../oled_sim
FAIL=0

echo "=== test 1: SCLC vs Mott-Gurney ==="
$SIM sclc_thick.ini > /dev/null
python3 - <<'EOF' || FAIL=1
import math
data=[]
for line in open('out_sclc_thick/device_figures.dat'):
    if line.startswith('#'): continue
    p=line.split()
    if len(p)>=2: data.append((float(p[0]),float(p[1])))
eps=3*8.8541878e-12; mu=1e-10; L=300e-9
V,J=data[-1]
mg=1.125*eps*mu*V*V/L**3
ratio=J/mg
V2,J2=data[-2]
slope=math.log(J/J2)/math.log(V/V2)
print(f'  V={V}: J/J_MG={ratio:.3f} (expect 1.0-1.15), log-log slope={slope:.3f} (expect >1.9)')
assert 1.0 < ratio < 1.15, 'SCLC ratio out of range'
assert slope > 1.9, 'SCLC slope too far from 2'
print('  PASS')
EOF

echo "=== test 2: JV vs gpvdm reference ==="
$SIM gpvdm_match.ini > /dev/null
python3 - <<'EOF' || FAIL=1
gp=[]
for line in open('reference_gpvdm_jv.csv'):
    if line.startswith('#'): continue
    p=line.replace(',',' ').split()
    if len(p)==2:
        try: gp.append((float(p[0]),float(p[1])))
        except ValueError: pass
mine={}
for line in open('out_gpvdm_match/device_figures.dat'):
    if line.startswith('#'): continue
    p=line.split()
    if len(p)>=2: mine[round(float(p[0]),4)]=float(p[1])
errs=[]
for v,j in gp:
    k=round(v,4)
    if k in mine and v>=0.4 and abs(j)>1e-4:
        errs.append(abs(mine[k]/j-1))
mean=sum(errs)/len(errs); worst=max(errs)
print(f'  {len(errs)} points V>=0.4: mean |err|={mean*100:.3f}%  max={worst*100:.3f}% (expect <1%)')
assert worst < 0.01, 'JV deviates from gpvdm reference by more than 1%'
print('  PASS')
EOF

echo "=== test 3: OLED example (convergence + confinement + outcoupling) ==="
(cd .. && ./oled_sim examples/oled.ini > /dev/null)
python3 - <<'EOF' || FAIL=1
blocks=[]; cur=[]; v=None
for line in open('../output_oled/device_profiles.dat'):
    if line.startswith('# V ='):
        if cur: blocks.append((v,cur))
        v=float(line.split('=')[1]); cur=[]
    elif line.strip() and not line.startswith('#'):
        cur.append([float(x) for x in line.split()])
if cur: blocks.append((v,cur))
assert blocks[-1][0] >= 3.99, f'sweep stopped early at V={blocks[-1][0]}'
rows=blocks[-1][1]
Rin=sum(r[5] for r in rows if 40e-9<=r[0]<=70e-9)
Rtot=sum(r[5] for r in rows)
frac=Rin/max(Rtot,1e-30)
print(f'  sweep completed to V={blocks[-1][0]}; recombination in EML: {frac*100:.1f}% (expect >95%)')
assert frac > 0.95, 'recombination not confined to the emissive layer'
eta=None
for line in open('../output_oled/outcoupling.dat'):
    if 'spectrum-averaged' in line:
        eta=float(line.split('eta_out=')[1].split()[0])
print(f'  outcoupling eta_out={eta} (expect 0<eta<1)')
assert eta is not None and 0.0 < eta < 1.0
# JV monotonic above turn-on
jv=[]
for line in open('../output_oled/device_figures.dat'):
    if line.startswith('#'): continue
    p=line.split()
    if len(p)>=2: jv.append((float(p[0]),float(p[1])))
above=[j for v,j in jv if v>=2.5]
assert all(b>a for a,b in zip(above,above[1:])), 'JV not monotonic above turn-on'
print('  PASS')
EOF

echo "=== test 4: Fermi-Dirac statistics (SCLC device) ==="
$SIM sclc.ini > /dev/null
$SIM sclc_fd.ini > /dev/null
python3 - <<'EOF' || FAIL=1
def load(p):
    d={}
    for line in open(p):
        if line.startswith('#'): continue
        t=line.split()
        if len(t)>=2: d[round(float(t[0]),3)]=float(t[1])
    return d
mb=load('out_sclc/device_figures.dat')
fd=load('out_sclc_fd/device_figures.dat')
v=5.0
r=fd[v]/mb[v]
print(f'  J_FD/J_MB at {v}V = {r:.4f} (expect ~1, within 5%: non-degenerate device)')
assert 0.95 < r < 1.05
print('  PASS')
EOF

echo "=== test 5: EGDM mobility OLED (convergence + physics) ==="
$SIM oled_egdm.ini > /dev/null 2>&1
python3 - <<'EOF' || FAIL=1
jv=[]
for line in open('output_oled_egdm/device_figures.dat'):
    if line.startswith('#'): continue
    p=line.split()
    if len(p)>=2: jv.append((float(p[0]),float(p[1])))
assert jv[-1][0] >= 3.49, f'EGDM sweep stopped early at V={jv[-1][0]}'
above=[j for v,j in jv if v>=2.0]
assert all(b>a for a,b in zip(above,above[1:])), 'EGDM JV not monotonic above turn-on'
print(f'  swept to V={jv[-1][0]}, J(3.5V)={jv[-1][1]:.3e} A/m^2, monotonic above 2V')
print('  PASS')
EOF

if [ "$FAIL" = "0" ]; then
	echo "=== all tests passed ==="
else
	echo "=== TESTS FAILED ==="
	exit 1
fi
