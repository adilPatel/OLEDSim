"""
training.py

The training loop: Adam with a step schedule over the weighted total loss,
the weights refreshed each epoch by whatever ``LossWeights`` rule the problem
holds.
"""

import torch

from .loss_weights import LOSS_TERMS, FixedWeights


def train(problem, *, epochs, n_int, lr=1e-3, milestones=None, gamma=0.3,
          log_every=20, print_every=2000, convergence_threshold=1e-10,
          loss_weights=None):
    """Train ``problem``, returning (epochs, losses) for the loss curve.

    epochs, n_int : number of steps, and collocation points sampled per step.
    lr, milestones, gamma : Adam learning rate and its MultiStepLR schedule.
    log_every : how often to record the loss for the curve.
    print_every : how often to print the per-term breakdown.
    convergence_threshold : stop early once the total loss falls below this.
    loss_weights : optional ``LossWeights`` to install for this run,
        replacing whatever the problem was built with. Its ``update`` runs
        every epoch; ``FixedWeights`` makes that a no-op, so there is no
        separate branch for a non-adaptive run.
    """
    if loss_weights is not None:
        problem.loss_weights = loss_weights
    weighting = problem.loss_weights

    # Adam owns the parameter list and applies the updates; it reads each
    # tensor's .grad, which backward() fills in below.
    optimizer = torch.optim.Adam(problem.all_params, lr=lr)
    # MultiStepLR multiplies the learning rate by gamma at each milestone
    # epoch, so training takes coarse steps early and fine ones later.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones or [], gamma=gamma)

    history_epochs = []
    device_losses = []          # kept on-device, drained in one sync at the end
    # The thermionic term is identically zero on a purely Ohmic device, so it
    # is only reported when the device has an injecting contact.
    has_thermionic = bool(problem.thermionic_contacts)
    # The verbose weight table needs a rule that records a trajectory.
    is_adaptive = not isinstance(weighting, FixedWeights)

    print("\n" + weighting.describe())
    print("\nTraining coupled drift-diffusion PINN (phi-Net + n-Net + p-Net)...")

    for epoch in range(epochs):
        # backward() ACCUMULATES into .grad rather than overwriting, so the
        # previous epoch's gradients must be cleared or they would compound.
        optimizer.zero_grad()
        loss, terms = problem.total_loss(n_int)

        # Before backward(), not after: an adaptive rule takes its own
        # autograd.grad over the same graph, which backward() would free. It
        # only reads gradients, so the step below is unaffected except through
        # the weights left behind.
        weighting.update(epoch, terms, problem.all_params)

        # backward() walks the graph from the scalar loss and writes d(loss)/
        # d(param) into each parameter's .grad; step() then applies them and
        # frees the graph. scheduler.step() advances the learning rate.
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            history_epochs.append(epoch)
            # detach() keeps the value on the accelerator but drops its graph,
            # so the history cannot pin thousands of epochs' graphs in memory.
            # Deliberately not .item(), which would sync the device every time.
            device_losses.append(loss.detach())

        if epoch % print_every == 0 or epoch == epochs - 1:
            # One column per term, in LOSS_TERMS order.
            reported = [t for t in LOSS_TERMS
                        if t != "thermionic" or has_thermionic]
            # .item() copies one value back to the CPU, forcing the device to
            # finish its queued work -- hence only on epochs that print.
            line = "  epoch {0:6d} | total {1:.3e}".format(epoch, loss.item())
            line += "".join(" | {0} {1:.3e}".format(t, terms[t].item())
                            for t in reported)
            line += " | lr {0:.1e}".format(scheduler.get_last_lr()[0])
            print(line)
            # Weights on their own lines, so the loss columns keep their
            # alignment and the weight trajectory reads down the log beside
            # the residuals it is balancing.
            if is_adaptive:
                print("         adaptive weights:")
                print(weighting.format_weights(verbose=True))

            if loss.item() < convergence_threshold:
                print("  Converged at epoch {0}, loss = {1:.3e}".format(
                    epoch, loss.item()))
                break

    # One transfer at the end instead of per epoch: stack the on-device
    # scalars into a tensor, .cpu() moves it to host memory, .numpy() views it
    # as an array (only legal once detached and on the CPU).
    history_losses = torch.stack(device_losses).cpu().numpy()
    return history_epochs, history_losses
