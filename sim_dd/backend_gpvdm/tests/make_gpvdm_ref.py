# make_gpvdm_ref.py - build a gpvdm v8.0 simulation directory that matches
# tests/gpvdm_match.ini, run gpvdm_core on it, and leave jv.dat for
# comparison.  Uses the gpvdm GUI's own json classes so the file carries
# every default the core expects.
#
#   python3 make_gpvdm_ref.py <output-dir>
#
# Device: single 100nm layer, gpvdm default material (Xi=1.6, Eg=1.2,
# eps_r=5, Nc=Nv=5e25, mue=muh=1e-5, B=0, exponential traps Nt=1e20,
# Et=60meV, 5 SRH bands, srh_start=-0.5, gpvdm default cross sections),
# ohmic contacts np=1e25: hole-majority at the top (anode, driven) and
# electron-majority at the bottom (cathode, ground).

import sys, os, json, subprocess

GPVDM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GUI = os.path.join(GPVDM_ROOT, "gpvdm_gui", "gui")
CORE = os.path.join(GPVDM_ROOT, "gpvdm_core", "gpvdm_core")

sys.path.insert(0, GUI)
os.chdir(GUI)  # gui modules resolve their data paths relative to here

from gpvdm_json import gpvdm_data

out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "gpvdm_ref")
os.makedirs(out, exist_ok=True)

d = gpvdm_data()
epi = d.epitaxy

epi.add_new_layer()          # single layer
epi.update_layer_type(0, "active")
layer = epi.layers[0]
layer.dy = 100e-9

# enable + configure the DOS of the active layer (gpvdm defaults, made explicit)
dos = layer.shape_dos
dos.enabled = True
dos.mue_y = 1e-5
dos.muh_y = 1e-5
dos.Nc = 5e25
dos.Nv = 5e25
dos.free_to_free_recombination = 0.0
dos.dostype = "exponential"
dos.Ntrape = 1e20
dos.Ntraph = 1e20
dos.Etrape = 60e-3
dos.Etraph = 60e-3
dos.srh_bands = 5
dos.srh_start = -0.5
dos.Xi = 1.6
dos.Eg = 1.2
dos.epsilonr = 5.0

epi.symc_to_mesh(d.mesh.mesh_y)
d.mesh.mesh_y.segments[0].points = 100
d.mesh.mesh_y.segments[0].mul = 1.0

c_top = epi.contacts.insert(0)
c_top.position = "top"
c_top.charge_type = "hole"
c_top.np = 1e25
c_top.physical_model = "ohmic"
c_top.applied_voltage_type = "change"
c_top.applied_voltage = 0.0

c_btm = epi.contacts.insert(1)
c_btm.position = "bottom"
c_btm.charge_type = "electron"
c_btm.np = 1e25
c_btm.physical_model = "ohmic"
c_btm.applied_voltage_type = "ground"
c_btm.applied_voltage = 0.0

from json_jv import json_jv_simulation
seg = json_jv_simulation()
seg.config.Vstart = 0.0
seg.config.Vstop = 2.0
seg.config.Vstep = 0.05
seg.config.jv_step_mul = 1.0
seg.config.jv_use_external_voltage_as_stop = False
d.jv.segments.append(seg)
d.sim.simmode = "segment0@jv"
d.light.Psun = 0.0

d.sim.version = "v8.0"
d.save_as(os.path.join(out, "sim.json"))
print("wrote", os.path.join(out, "sim.json"))

r = subprocess.run([CORE], cwd=out, capture_output=True, text=True, timeout=600)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("gpvdm_core exited", r.returncode)
    print(r.stderr[-2000:])
jv = os.path.join(out, "sim", "jv.dat")
if os.path.exists(jv):
    print("reference JV written:", jv)
else:
    print("look for jv output under", out)
