"""
Turn a solver-independent Device mesh into a DEVSIM 1-D device.

The mesh *positions* and contact definitions live on ``core.device.Device`` and
know nothing about DEVSIM. This module is the DEVSIM-specific step that consumes
that plain data and issues the ``create_1d_mesh`` / ``add_1d_*`` / ``create_device``
calls. A different back end would provide its own equivalent of this file.
"""

from devsim import (
    create_1d_mesh,
    add_1d_mesh_line,
    add_1d_contact,
    add_1d_region,
    finalize_mesh,
    create_device,
)


def create_mesh(device_name, region, device, mesh_name="oled"):
    """Build the DEVSIM 1-D mesh/device from ``device`` (a core.device.Device).

    Reads ``device.mesh_positions_nm`` (a sorted float array of node positions
    in nm) and ``device.contacts`` (top/bottom contact definitions). The mesh
    spacing at each node is taken from the gap to the next node, matching the
    original oled1.py behaviour.
    """
    create_1d_mesh(mesh=mesh_name)

    positions_nm = device.mesh_positions_nm
    nm_to_cm = 1e-7
    n = len(positions_nm)

    for i, pos_nm in enumerate(positions_nm):
        if i == 0:
            tag = device.top_contact.tag
        elif i == n - 1:
            tag = device.bottom_contact.tag
        else:
            tag = None

        if i < n - 1:
            ps = (positions_nm[i + 1] - pos_nm) * nm_to_cm
        else:
            ps = (pos_nm - positions_nm[i - 1]) * nm_to_cm

        if tag is not None:
            add_1d_mesh_line(mesh=mesh_name, pos=pos_nm * nm_to_cm, ps=ps, tag=tag)
        else:
            add_1d_mesh_line(mesh=mesh_name, pos=pos_nm * nm_to_cm, ps=ps)

    top = device.top_contact
    bot = device.bottom_contact
    add_1d_contact(mesh=mesh_name, name=top.name, tag=top.tag, material=top.material)
    add_1d_contact(mesh=mesh_name, name=bot.name, tag=bot.tag, material=bot.material)
    add_1d_region(mesh=mesh_name, material=device.material, region=region,
                  tag1=top.tag, tag2=bot.tag)
    finalize_mesh(mesh=mesh_name)
    create_device(mesh=mesh_name, device=device_name)
