"""
loss_weights.py

The multipliers on the loss terms, and the rules that produce them.

``LossWeights`` is the interface the training loop holds: a live ``weights``
dict that ``PINNProblem.total_loss`` reads each epoch, and an ``update`` hook
called before ``backward()``. Two rules implement it -- ``FixedWeights``,
whose update is a no-op, and ``InverseDirichletWeights``, which resets the
weights from the gradient balance as training proceeds.
"""

import torch

# Term order used everywhere the weights are handled as a group, fixed here so
# printed columns keep a stable order.
LOSS_TERMS = ("poisson", "cont_n", "cont_p", "jtot", "bc", "thermionic")

# Hand-set defaults. The two continuity residuals get separate weights, and
# Poisson is weighted well above the rest since it is the hardest term to
# converge. `thermionic` at 1.0 puts the flux balance on the same footing as
# the Dirichlet pins it replaces; it is a no-op on a purely Ohmic device,
# where the term is identically zero.
DEFAULT_LOSS_WEIGHTS = {
    "bc": 1.0,
    "cont_n": 1.0,
    "cont_p": 1.0e3,
    "jtot": 1.0,
    "poisson": 100.0,
    "thermionic": 1.0,
}


class LossWeights:
    """Base class: a live weight dict plus an update hook.

    ``self.weights`` is mutated in place rather than rebound, so
    ``PINNProblem.total_loss`` can hold the same dict for the whole run and
    pick up new values without any further plumbing.
    """

    def __init__(self, initial_weights):
        self.weights = dict(initial_weights)

    def update(self, epoch, loss_terms, params):
        """Recompute the weights. Returns True when they actually changed.

        loss_terms : dict of term name -> unweighted loss tensor.
        params : the shared parameters gradients are taken w.r.t.
        """
        return False

    def describe(self):
        """One-block summary printed at the start of training."""
        return "Loss weighting: fixed\n  " + self.format_weights()

    def format_weights(self, terms=None, verbose=False):
        """One-line `name value` summary of the live weights."""
        terms = list(self.weights) if terms is None else terms
        return " ".join("{0} {1:.3e}".format(t, self.weights[t]) for t in terms)


class FixedWeights(LossWeights):
    """Hand-set weights, held constant for the whole run.

    Inherits the no-op ``update``, so the training loop needs no branch for
    the non-adaptive case.
    """


class InverseDirichletWeights(LossWeights):
    """Inverse-Dirichlet loss weighting (Maddu, Sturm, Muller & Sbalzarini,
    *Machine Learning: Science and Technology* 3 (2022) 015026).

    The problem it solves: the loss terms are residuals of different equations
    in different units, so their gradients w.r.t. the shared parameters can
    differ by orders of magnitude, and whichever term carries the largest
    gradient dominates the update. A hand-set weight fixes this only for the
    operating point it was tuned at.

    The rule sets every term's weight from the *standard deviation* of its
    gradient distribution, so the weighted gradients have comparable spread:

        lambda_hat_i = std_max / std(grad_theta L_i),   std_max = max_j std_j

    relaxed into the live weight as
    lambda_i <- (1 - alpha)*lambda_i + alpha*lambda_hat_i.

    Why std and not mean -- this is what distinguishes the scheme from a
    gradient-magnitude ratio. A mean denominator cannot separate a *converged*
    term (gradients small and uniform, nothing left to fix) from a *stiff* one
    (small on average but very unevenly distributed). Both look like "small
    mean gradient", so a mean-based rule up-weights both and the converged one
    wins, because its gradient goes to zero furthest. The standard deviation
    measures how *unbalanced* a term's gradient field is, which is what "still
    has work to do" looks like. That matters concretely here: every term
    except Poisson admits a trivial minimiser (a pin met exactly, a variance
    zeroed by any constant current, a flat density profile), so a mean-based
    rule hands the weight budget to whichever term is nearest a trivial zero
    and starves the one term that cannot be trivially satisfied.

    What bounds it: the numerator is a maximum *over the terms themselves*, so
    the weights track the imbalance actually present rather than an external
    reference. The largest-spread term always sits at lambda = 1 and the
    others are raised relative to it. This does not bound the weights
    absolutely -- a term whose gradient spread is orders of magnitude below
    the largest gets a correspondingly large weight -- but a weight grows only
    in proportion to the spread deficit it is derived from, and no term is
    suppressed to fund another.

    Like any gradient-statistics scheme this balances *trainability*, not
    accuracy: it equalises how strongly each term pulls on the shared
    parameters, and does not know which term matters physically. It is a
    starting point to be scored against a reference solution.
    """

    def __init__(self, *, initial_weights, alpha=0.5, update_every=1,
                 terms=None, reference=None, normalize=False, max_ratio=None,
                 eps=1e-16):
        """
        initial_weights : starting weights. Adapted terms are overwritten as
            training proceeds; the rest are held at their given value.
        alpha : running-average rate. Larger than a magnitude-ratio scheme
            would use (0.5 against 0.1) because the std statistic is far less
            noisy, so the weights can track it without chasing collocation
            noise.
        update_every : recompute every N epochs. Each update costs one
            backward pass per adapted term.
        terms : which terms adapt; defaults to every term in
            initial_weights. Unlike a reference-anchored scheme, all of them
            can -- the numerator is a max over the terms rather than one
            chosen term's gradient.
        reference : optional term excluded from adaptation and held at its
            initial weight. Normally None; provided only so a run can pin one
            equation deliberately.
        normalize : renormalise the adapted weights to sum to their count.
            **Default False, which is the paper's formulation.** Unnormalised,
            the weights are *absolute* amplifications: the largest-spread term
            keeps 1.0 and the others are raised to match it. Renormalising
            turns them into shares of a fixed budget, so a term with a large
            target divides every other weight down -- including that of the
            term setting the reference scale. That is a different algorithm,
            not a tidying step.
        max_ratio : optional cap on a single term's target ratio. **Not part
            of the published method -- a local addition, default off.** A
            diagnostic escape hatch rather than a tuning knob: a term whose
            gradient spread is small *in proportion to its own loss* is
            physically negligible rather than stiff, and the std criterion
            cannot tell those apart from gradients alone, so a term hitting
            the cap is a finding about the device. ``capped`` records which
            terms hit it.
        eps : floor below which a gradient std is treated as absent and the
            term skipped. Used **only** as that threshold, deliberately not
            added to the division in step 3 -- step 1 has already guaranteed
            std > eps there, and adding it would perturb the ratio for a term
            with a very small spread.
        """
        super().__init__(initial_weights)
        self.alpha = alpha
        self.update_every = update_every
        self.reference = reference
        self.normalize = normalize
        self.max_ratio = max_ratio
        self.eps = eps
        # Terms whose target hit max_ratio, reported so a physically-scaled
        # term (as opposed to a stiff one) is visible rather than silent.
        self.capped = set()
        if terms is None:
            terms = [t for t in LOSS_TERMS
                     if t in self.weights and t != reference]
        self.terms = list(terms)
        # Per-term history, sampled whenever update() runs. The instantaneous
        # target and the gradient std that produced it are kept alongside the
        # smoothed weight, because the gap between them shows how hard the
        # weight is still moving.
        self.history_epochs = []
        self.history = {t: [] for t in self.terms}
        self.history_target = {t: [] for t in self.terms}
        self.history_std = {t: [] for t in self.terms}

    def _grad_std(self, loss_term, params):
        """Standard deviation of one loss term's gradient w.r.t. params.

        retain_graph=True because every term shares the one forward graph;
        allow_unused because a term need not touch every network (the density
        pins do not involve phi-Net, for instance).
        """
        grads = torch.autograd.grad(
            loss_term, params, retain_graph=True, allow_unused=True)
        parts = [g.reshape(-1) for g in grads if g is not None]
        if not parts:
            return None
        flat = torch.cat(parts)
        if flat.numel() < 2:
            return None
        return flat.std()

    def update(self, epoch, loss_terms, params):
        """Recompute the weights from the current gradient spreads."""
        # Step 0: throttle. Each update costs one backward pass per term.
        if epoch % self.update_every != 0:
            return False

        # Step 1: gradient std per adapted term, in one pass over the shared
        # graph. Terms carrying no usable signal keep their current weight.
        stds = {}
        for t in self.terms:
            L = loss_terms.get(t)
            if self._is_inactive(L):
                continue
            sd = self._grad_std(L, params)
            # No detach() needed: autograd.grad() without create_graph=True
            # returns gradients already outside the graph, so `sd` has
            # requires_grad=False and float() neither warns nor differs.
            if sd is None or float(sd) <= self.eps:
                continue
            stds[t] = float(sd)

        if not stds:
            return False

        # Step 2: the numerator is the largest spread among the terms
        # themselves, so the ratio is bounded by the imbalance in the loss.
        std_max = max(stds.values())

        # Step 3: target ratio, relaxed into the live weight. No epsilon guard
        # on the division -- step 1 already dropped every term with
        # sd <= eps, so sd is strictly positive here.
        targets = {}
        for t, sd in stds.items():
            lambda_hat = std_max / sd
            if self.max_ratio is not None and lambda_hat > self.max_ratio:
                lambda_hat = self.max_ratio
                self.capped.add(t)
            targets[t] = lambda_hat
            self.weights[t] = ((1.0 - self.alpha) * self.weights[t]
                               + self.alpha * lambda_hat)

        # Step 4: optional renormalisation to a fixed budget (off by default;
        # see the class docstring for why).
        if self.normalize:
            active = [t for t in self.terms if t in stds]
            total = sum(self.weights[t] for t in active)
            if total > self.eps:
                scale = len(active) / total
                for t in active:
                    self.weights[t] *= scale

        # Step 5: record the trajectory. A term skipped this round records NaN
        # rather than a stale value, so a gap in the trace is visibly a gap.
        self.history_epochs.append(epoch)
        for t in self.terms:
            self.history[t].append(self.weights[t])
            self.history_target[t].append(targets.get(t, float("nan")))
            self.history_std[t].append(stds.get(t, float("nan")))
        return True

    def growth(self, t):
        """Multiplicative change in weight `t` over the last two updates.

        >1 rising, <1 falling, 1.0 static. Reported per update rather than per
        epoch, so it is comparable between runs with different update_every.
        """
        h = self.history[t]
        if len(h) < 2 or h[-2] == 0.0:
            return float("nan")
        return h[-1] / h[-2]

    def describe(self):
        lines = ["Adaptive loss weighting: inverse-Dirichlet "
                 "(Maddu, Sturm, Muller & Sbalzarini 2022)"]
        if self.reference is not None:
            lines.append("  held fixed     : {0} (weight {1:.3e})".format(
                self.reference, self.weights[self.reference]))
        lines.append("  alpha          : {0}   update every {1} epoch(s)".format(
            self.alpha, self.update_every))
        lines.append("  normalised     : {0}".format(
            "yes, weights sum to {0}".format(len(self.terms))
            if self.normalize else "no"))
        lines.append("  adapting       : {0}".format(", ".join(self.terms)))
        return "\n".join(lines)

    def format_weights(self, terms=None, verbose=False):
        """Summary of the live weights for the training log.

        verbose=True gives, per term, the weight, its per-update growth factor
        with a direction arrow, the instantaneous target before smoothing, and
        the gradient std that produced it. A weight that is *settling* has
        growth -> 1 with the target close to the weight; one still moving has
        the two apart. With normalize off the weights are absolute
        amplifications rather than shares, so one rising does not imply
        another falling.
        """
        terms = self.terms if terms is None else terms
        if not verbose:
            return super().format_weights(terms)

        lines = ["    {0:<11s} {1:>10s} {2:>9s}  {3:>10s} {4:>10s}".format(
            "term", "weight", "growth", "target", "std|grad|")]
        for t in terms:
            g = self.growth(t)
            if g != g:                       # NaN
                arrow, gtxt = " ", "    --   "
            elif g > 1.05:
                arrow, gtxt = "^", "{0:8.3f}x".format(g)
            elif g < 0.95:
                arrow, gtxt = "v", "{0:8.3f}x".format(g)
            else:
                arrow, gtxt = "=", "{0:8.3f}x".format(g)
            tgt = self.history_target[t][-1] if self.history_target[t] else float("nan")
            sd = self.history_std[t][-1] if self.history_std[t] else float("nan")
            cap = " (CAPPED)" if t in self.capped else ""
            lines.append("    {0:<11s} {1:10.3e} {2}{3}  {4:10.3e} {5:10.3e}{6}".format(
                t, self.weights[t], arrow, gtxt, tgt, sd, cap))
        return "\n".join(lines)

    @staticmethod
    def _is_inactive(L):
        """True for a term carrying no usable gradient signal.

        Three cases, all meaning "do not adapt this weight": absent from the
        loss dict; detached from the graph (how a hard boundary ansatz
        presents -- the boundary loss is a literal zero tensor then); or
        identically zero (thermionic on a purely Ohmic device).

        The .detach() is cosmetic -- float() on a tensor builds no graph node
        either way. It only suppresses the UserWarning float() raises on a
        tensor with requires_grad=True, and marks the read as control flow.
        (Contrast the detach in poisson_residual, which is load-bearing.)
        """
        return (L is None
                or not L.requires_grad
                or float(L.detach()) == 0.0)


def make_weights(kind="fixed", *, initial_weights=None, **kwargs):
    """Build a LossWeights of the requested kind.

    kind : "fixed" or "inverse_dirichlet". Extra keyword arguments are passed
        to InverseDirichletWeights and ignored by the fixed rule.
    """
    weights = DEFAULT_LOSS_WEIGHTS if initial_weights is None else initial_weights
    if kind == "fixed":
        return FixedWeights(weights)
    if kind == "inverse_dirichlet":
        return InverseDirichletWeights(initial_weights=weights, **kwargs)
    raise ValueError(
        "unknown loss-weight kind {0!r}; expected 'fixed' or "
        "'inverse_dirichlet'".format(kind))
