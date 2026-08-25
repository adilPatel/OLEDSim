"""
Physics-informed neural-network (PINN) drift-diffusion solver.

``core`` holds the device-independent model -- the three subnetworks, the
scaling, the residuals and the training loop. ``devices`` holds one script per
device, supplying its parameters, boundary conditions and run configuration,
and scoring the result against a DEVSIM reference.
"""
