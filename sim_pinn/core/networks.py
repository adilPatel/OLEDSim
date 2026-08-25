"""
networks.py

The backbone shared by all three subnetworks, and the autodiff helper the
residuals are built from.

phi-Net (poisson.py) and n-Net/p-Net (densities.py) use the same architecture;
what distinguishes them is the output parametrisation and boundary ansatz each
wraps around it.
"""

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """Fully-connected tanh network mapping x -> a single scalar output.

    tanh rather than ReLU: the Poisson residual takes a second derivative
    through autodiff, which a piecewise-linear activation cannot supply.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Xavier/Glorot initialization.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Map [0, 1] -> [-1, 1]: tanh is centred there.
        return self.net(2.0 * x - 1.0)


def d_dx(f, x):
    """First derivative df/dx via automatic differentiation.

    ``create_graph=True`` keeps the derivative itself differentiable, needed
    for Poisson's second derivative and for backpropagating through any
    residual that contains a gradient.
    """
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
