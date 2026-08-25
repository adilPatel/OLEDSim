"""
training.py

The training loop: Adam with a step schedule over the weighted total loss,
with the loss weights refreshed each epoch by whatever ``LossWeights`` rule
the problem holds.
"""

import torch

from .loss_weights import LOSS_TERMS, InverseDirichletWeights


def train(problem, *, epochs, n_int, lr=1e-3, milestones=None, gamma=0.3,
          log_every=20, print_every=2000, convergence_threshold=1e-10,
          loss_weights=None):
    """Run the training loop, returning (epochs, losses) for the loss curve.

    loss_weights : optional ``LossWeights`` to install on the problem for this
        run, replacing whatever it was built with. Its ``update`` is called
        every epoch; ``FixedWeights`` makes that a no-op, so there is no
        separate branch for a non-adaptive run.
    """
    if loss_weights is not None:
        problem.loss_weights = loss_weights
    weighting = problem.loss_weights

    optimizer = torch.optim.Adam(problem.all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones or [], gamma=gamma)

    history_epochs = []
    device_losses = []          # kept on-device, drained in one sync at the end
    # The thermionic term is identically zero for a purely Ohmic device, so it
    # is only reported when the device actually has an injecting contact.
    has_thermionic = bool(problem.thermionic_contacts)
    # Verbose per-term weight reporting only means something for a rule that
    # records a trajectory.
    is_adaptive = isinstance(weighting, InverseDirichletWeights)

    print("\n" + weighting.describe())
    print("\nTraining coupled drift-diffusion PINN (phi-Net + n-Net + p-Net)...")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, terms = problem.total_loss(n_int)

        # The weight update has to run *before* backward(): it takes its own
        # autograd.grad over the same graph, which backward() would free. It
        # only reads gradients -- the parameter update below is unaffected
        # except through the weights it leaves behind.
        weighting.update(epoch, terms, problem.all_params)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            history_epochs.append(epoch)
            device_losses.append(loss.detach())

        if epoch % print_every == 0 or epoch == epochs - 1:
            reported = [t for t in LOSS_TERMS
                        if t != "thermionic" or has_thermionic]
            line = "  epoch {0:6d} | total {1:.3e}".format(epoch, loss.item())
            line += "".join(" | {0} {1:.3e}".format(t, terms[t].item())
                            for t in reported)
            line += " | lr {0:.1e}".format(scheduler.get_last_lr()[0])
            print(line)
            # The live weights on their own line, so the loss columns keep
            # their alignment and the weight trajectory can be read down the
            # log next to the residuals it is balancing.
            if is_adaptive:
                print("         adaptive weights:")
                print(weighting.format_weights(verbose=True))

            if loss.item() < convergence_threshold:
                print("  Converged at epoch {0}, loss = {1:.3e}".format(
                    epoch, loss.item()))
                break

    history_losses = torch.stack(device_losses).cpu().numpy()
    return history_epochs, history_losses
