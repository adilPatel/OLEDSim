from devsim import *
create_1d_mesh(mesh="m")
add_1d_mesh_line(mesh="m", pos=0, ps=1e-7, tag="top")
add_1d_mesh_line(mesh="m", pos=1e-5, ps=1e-7, tag="bot")
add_1d_contact(mesh="m", name="top", tag="top", material="metal")
add_1d_contact(mesh="m", name="bot", tag="bot", material="metal")
add_1d_region(mesh="m", material="Silicon", region="r", tag1="top", tag2="bot")
finalize_mesh(mesh="m")
create_device(mesh="m", device="d")
node_solution(name="Potential", device="d", region="r")
edge_from_node_model(node_model="Potential", device="d", region="r")
edge_average_model(device="d", region="r", node_model="Potential", edge_model="EField", average_type="negative_gradient")
edge_average_model(device="d", region="r", node_model="Potential", edge_model="EField", average_type="negative_gradient", derivative="Potential")
n = len(get_node_model_values(device="d", region="r", name="Potential"))
set_node_values(device="d", region="r", name="Potential", values=[0.0]*n)

# smooth regularization: EFieldMag = sqrt(EField^2 + eps^2) -- always > 0, smooth, differentiable everywhere
eps = 1.0
expr = "((EField*EField) + {0})^0.5".format(eps*eps)
contact_edge_model(device="d", contact="top", name="EFieldMag", equation=expr)
contact_edge_model(device="d", contact="top", name="EFieldMag:Potential@n0", equation="diff({0},Potential@n0)".format(expr))
print("EFieldMag at EField=0:", get_edge_model_values(device="d", region="r", name="EFieldMag")[0], flush=True)
print("d(EFieldMag)/dPot@n0 at EField=0:", get_edge_model_values(device="d", region="r", name="EFieldMag:Potential@n0")[0], flush=True)
print("NO CRASH - regularization works", flush=True)
