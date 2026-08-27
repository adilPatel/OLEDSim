"""
networks.py

The backbone shared by all three subnetworks, and the autodiff helper the
residuals are built from.

phi-Net (poisson.py) and n-Net/p-Net (densities.py) wrap this same
architecture; what distinguishes them is the output parametrisation and
boundary ansatz each applies to it.
"""

import torch
import torch.nn as nn


# nn.Module is torch's base class for anything holding trainable parameters:
# it tracks them for the optimizer and moves them together on .to(device).
class FCNN(nn.Module):
    """Fully-connected tanh network mapping x -> one scalar output.

    width, depth : hidden width and number of hidden layers.

    tanh rather than ReLU: the Poisson residual takes a second derivative
    through autodiff, which a piecewise-linear activation cannot supply.
    """

    def __init__(self, width=64, depth=4):
        super().__init__()
        # depth hidden layers: the first maps 1 -> width, the rest width ->
        # width, then a linear head back down to 1. nn.Linear is an affine
        # layer Wx + b; nn.Tanh the activation between them.
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        # nn.Sequential chains them into one callable, so forward() is a
        # single call and the parameters of every layer are registered here.
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                # Xavier/Glorot init. The trailing underscore marks a torch
                # in-place operation: it overwrites the tensor rather than
                # returning a new one.
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # forward() defines what the module computes; torch calls it when the
    # module is invoked as net(x).
    def forward(self, x):
        # Map the domain [0, 1] -> [-1, 1], where tanh is centred.
        return self.net(2.0 * x - 1.0)


def d_dx(f, x):
    """First derivative df/dx by automatic differentiation.

    torch.autograd.grad differentiates f with respect to x by replaying the
    operations recorded when f was computed -- exact, not a finite difference.
    It returns a tuple (one entry per input), hence the [0].

    grad_outputs=ones is the seed vector: autograd computes a vector-Jacobian
    product, and seeding with ones over a batch of independent points gives
    each point its own df/dx.

    create_graph=True records the derivative computation too, so the result is
    itself differentiable -- needed for Poisson's second derivative, and to
    backpropagate through any residual containing a gradient.
    """
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
