"""
loss_weights.py

The multipliers on the loss terms, and the rules that produce them.

``LossWeights`` is the interface the training loop holds: a live ``weights``
dict that ``PINNProblem.total_loss`` reads each epoch, and an ``update`` hook
called before ``backward()``. Three rules implement it -- ``FixedWeights``
(no-op update), ``InverseDirichletWeights`` (weights from the gradient
balance) and ``SoftAdaptWeights`` (weights from the loss rates of change).
"""

import math

import torch

# Term order used wherever the weights are handled as a group, fixed here so
# printed columns keep a stable order.
LOSS_TERMS = ("poisson", "cont_n", "cont_p", "jtot", "bc", "thermionic")

# Hand-set defaults. The two continuity residuals get separate weights, and
# Poisson is weighted well above the rest since it is the hardest term to
# converge. `thermionic` at 1.0 puts the flux balance on the same footing as
# the Dirichlet pins it replaces; it is a no-op on a purely Ohmic device.
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
    ``PINNProblem.total_loss`` can hold the same dict for a whole run and pick
    up new values without further plumbing.
    """

    def __init__(self, initial_weights):
        """initial_weights : dict of term -> weight, copied into self.weights."""
        self.weights = dict(initial_weights)

    def update(self, epoch, loss_terms, params):
        """Recompute the weights. Returns True when they actually changed.

        epoch : current epoch, for rules that update on a schedule.
        loss_terms : dict of term -> unweighted loss tensor.
        params : shared parameters the gradients are taken w.r.t.
        """
        return False

    def describe(self):
        """One-block summary printed at the start of training."""
        return "Loss weighting: fixed\n  " + self.format_weights()

    def growth(self, t):
        """Multiplicative change in term t's weight over the last two updates.

        >1 rising, <1 falling, 1.0 static; NaN before two updates, or if the
        rule keeps no history. Per update rather than per epoch, so it is
        comparable between runs with different update_every.
        """
        h = getattr(self, "history", {}).get(t, [])
        if len(h) < 2 or h[-2] == 0.0:
            return float("nan")
        return h[-1] / h[-2]

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

    The loss terms are residuals of different equations in different units, so
    their gradients w.r.t. the shared parameters can differ by orders of
    magnitude and the largest dominates the update. This sets each weight from
    the *standard deviation* of that term's gradient distribution, so the
    weighted gradients have comparable spread:

        lambda_hat_i = std_max / std(grad_theta L_i),   std_max = max_j std_j
        lambda_i    <- (1 - alpha)*lambda_i + alpha*lambda_hat_i

    Why std and not mean, which is what separates this from a
    gradient-magnitude ratio: a mean denominator cannot distinguish a
    *converged* term (gradients small and uniform) from a *stiff* one (small
    on average, very unevenly distributed). Both look like "small mean
    gradient", so a mean-based rule up-weights both and the converged one wins.
    That matters here because every term except Poisson admits a trivial
    minimiser -- a pin met exactly, a variance zeroed by any constant current,
    a flat density profile -- so such a rule would starve the one term that
    cannot be trivially satisfied.

    The numerator is a maximum *over the terms themselves*, so the weights
    track the imbalance actually present: the largest-spread term sits at
    lambda = 1 and the others are raised relative to it. That does not bound
    them absolutely, but a weight grows only in proportion to its own spread
    deficit and no term is suppressed to fund another.

    Like any gradient-statistics scheme this balances *trainability*, not
    accuracy: it does not know which term matters physically, so it is a
    starting point to be scored against a reference.
    """

    scheme_name = "Inverse-Dirichlet"

    def __init__(self, *, initial_weights, alpha=0.5, update_every=1,
                 terms=None, reference=None, normalize=False, max_ratio=None,
                 eps=1e-16):
        """
        initial_weights : starting weights. Adapted terms are overwritten as
            training proceeds; the rest are held.
        alpha : running-average rate. Higher than a magnitude-ratio scheme
            would use (0.5 against 0.1), the std statistic being far less
            noisy, so the weights track it without chasing collocation noise.
        update_every : recompute every N epochs; each update costs one
            backward pass per adapted term.
        terms : which terms adapt. Default every term in initial_weights --
            unlike a reference-anchored scheme, all of them can, the numerator
            being a max over the terms rather than one chosen gradient.
        reference : optional term excluded from adaptation and held at its
            initial weight. Normally None; only for pinning one equation
            deliberately.
        normalize : renormalise the adapted weights to sum to their count.
            **Default False, the paper's formulation.** Unnormalised the
            weights are *absolute* amplifications, the largest-spread term
            keeping 1.0 and the rest raised to match. Renormalising makes them
            shares of a fixed budget, so one large target divides every other
            weight down -- a different algorithm, not a tidying step.
        max_ratio : optional cap on a term's target ratio. **Not part of the
            published method.** A diagnostic rather than a tuning knob: a term
            whose spread is small *in proportion to its own loss* is physically
            negligible rather than stiff, and the std criterion cannot tell
            those apart, so hitting the cap is a finding about the device.
            ``capped`` records which terms did.
        eps : floor below which a gradient std counts as absent and the term is
            skipped. Used **only** as that threshold, deliberately not added to
            the ratio's divisor, which is already guaranteed positive.
        """
        super().__init__(initial_weights)
        self.alpha = alpha
        self.update_every = update_every
        self.reference = reference
        self.normalize = normalize
        self.max_ratio = max_ratio
        self.eps = eps
        # Terms whose target hit max_ratio, recorded so a physically-scaled
        # term (as opposed to a stiff one) is visible rather than silent.
        self.capped = set()
        if terms is None:
            terms = [t for t in LOSS_TERMS
                     if t in self.weights and t != reference]
        self.terms = list(terms)
        # Per-term trajectory, appended once per update. The pre-smoothing
        # target and the std behind it sit alongside the live weight, since
        # the gap between them shows how hard the weight is still moving.
        self.history_epochs = []
        self.history = {t: [] for t in self.terms}          # live lambda_i
        self.history_target = {t: [] for t in self.terms}   # lambda_hat_i
        self.history_std = {t: [] for t in self.terms}      # std_i
        self.history_loss = {t: [] for t in self.terms}     # unweighted L_i

    def _grad_std(self, loss_term, params):
        """Standard deviation of one term's gradient w.r.t. params, or None if
        it has no usable gradient.

        retain_graph=True because every term shares the one forward graph;
        allow_unused because a term need not touch every network -- the
        density pins do not involve phi-Net, for instance.
        """
        # autograd.grad returns this term's gradient w.r.t. each parameter
        # tensor, WITHOUT accumulating into .grad the way backward() would --
        # so calling it here leaves the optimizer's own gradients untouched.
        #
        # retain_graph=True: the graph is normally freed after the first
        # traversal, and every term shares this one forward pass, so it must
        # survive for the remaining terms and for the real backward() after.
        # allow_unused=True: a term need not touch every network (the density
        # pins do not involve phi-Net), and torch would otherwise raise rather
        # than return None for those.
        grads = torch.autograd.grad(
            loss_term, params, retain_graph=True, allow_unused=True)
        # reshape(-1) flattens each parameter's gradient to 1-D, and torch.cat
        # joins them, so the std below is taken over ALL parameters at once
        # rather than per layer.
        parts = [g.reshape(-1) for g in grads if g is not None]
        if not parts:
            return None
        flat = torch.cat(parts)
        if flat.numel() < 2:          # numel = element count; std of one is undefined
            return None
        return flat.std()

    def update(self, epoch, loss_terms, params):
        """Recompute the weights from the current gradient spreads.

        epoch : current epoch, used for the throttle.
        loss_terms : dict of term -> unweighted loss tensor.
        params : shared parameters the gradients are taken w.r.t.

        Returns True when the weights changed.
        """
        # Throttle: each update costs one backward pass per adapted term.
        if epoch % self.update_every != 0:
            return False

        # Step 1: per-term gradient spread, plus the loss itself for plotting.
        # A term with no usable gradient is left out and keeps its weight.
        stds = {}
        losses = {}
        for t in self.terms:
            L = loss_terms.get(t)
            if self._is_inactive(L):
                continue
            # detach() before float(): the value is only being recorded, and
            # reading a graph-attached tensor as a float raises a UserWarning.
            losses[t] = float(L.detach())
            sd = self._grad_std(L, params)
            # No detach needed here: autograd.grad without create_graph=True
            # returns gradients already outside the graph, so `sd` carries no
            # requires_grad and float() neither warns nor changes the value.
            if sd is None or float(sd) <= self.eps:
                continue
            stds[t] = float(sd)

        if not stds:
            return False

        # Step 2: the reference is the largest spread among the terms
        # themselves, so the ratio is bounded by the imbalance really present.
        std_max = max(stds.values())

        # Step 3: target ratio per term, relaxed into the live weight. No
        # epsilon on the division -- step 1 dropped every sd <= eps.
        targets = {}
        for t, sd in stds.items():
            lambda_hat = std_max / sd
            if self.max_ratio is not None and lambda_hat > self.max_ratio:
                lambda_hat = self.max_ratio
                self.capped.add(t)
            targets[t] = lambda_hat
            self.weights[t] = ((1.0 - self.alpha) * self.weights[t]
                               + self.alpha * lambda_hat)

        # Step 4: optional rescale to a fixed budget (off by default; see the
        # class docstring).
        if self.normalize:
            active = [t for t in self.terms if t in stds]
            total = sum(self.weights[t] for t in active)
            if total > self.eps:
                scale = len(active) / total
                for t in active:
                    self.weights[t] *= scale

        # Step 5: record the trajectory. A term skipped this round stores NaN
        # rather than a stale value, so a gap in a trace reads as a gap.
        self.history_epochs.append(epoch)
        for t in self.terms:
            self.history[t].append(self.weights[t])
            self.history_target[t].append(targets.get(t, float("nan")))
            self.history_std[t].append(stds.get(t, float("nan")))
            self.history_loss[t].append(losses.get(t, float("nan")))
        return True

    def describe(self):
        """One-block summary of the rule and its settings, for the run log."""
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

        verbose=True adds, per term, the per-update growth factor with a
        direction arrow, the pre-smoothing target and the gradient std behind
        it. A settling weight has growth -> 1 with target close to weight; one
        still moving has the two apart. With normalize off the weights are
        absolute amplifications, so one rising does not imply another falling.
        """
        terms = self.terms if terms is None else terms
        if not verbose:
            return super().format_weights(terms)

        lines = ["    {0:<11s} {1:>10s} {2:>9s}  {3:>10s} {4:>10s}".format(
            "term", "weight", "growth", "target", "std|grad|")]
        for t in terms:
            g = self.growth(t)
            if g != g:                       # NaN: fewer than two updates
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
        """True for a term carrying no usable gradient signal: absent from the
        loss dict, detached from the graph (how a hard boundary ansatz
        presents -- the boundary loss is then a literal zero tensor), or
        identically zero (thermionic on a purely Ohmic device).

        The detach() is cosmetic: float() builds no graph node either way, so
        it only suppresses the UserWarning on a requires_grad tensor and marks
        the read as control flow. Contrast the detach in poisson_residual,
        which is load-bearing.
        """
        # requires_grad is False when a tensor is not connected to any
        # trainable parameter -- the case for a literal zeros() placeholder.
        return (L is None
                or not L.requires_grad
                or float(L.detach()) == 0.0)


class SoftAdaptWeights(LossWeights):
    """SoftAdapt loss weighting (Heydari, Thompson & Mehmood, arXiv:1912.12355).

    Weights each term by its recent *relative* rate of change, through a
    loss-weighted softmax:

        s_i      = ( mean L_i(t) - mean L_i(t-1) ) / |mean L_i(t-1)|
        f_i      = mean L_i(t) / sum_j mean L_j(t)
        lambda_i = f_i*exp(beta*s_i) / sum_j f_j*exp(beta*s_j)

    where the means are over the update_every epochs since the last update.
    beta > 0 favours slowly-improving terms; f_i keeps a small, noisy term
    from capturing the budget, since s_i is unbounded above.

    The weights are shares: lambda_i in (0, 1) and sum_i lambda_i = 1.

    Rationale for the relative rate, the loss weighting and the max-shift in
    _softmax is in Models_neural.md, "Adaptive loss weights - SoftAdapt".
    """

    scheme_name = "SoftAdapt"

    def __init__(self, *, initial_weights, beta=1.0, update_every=1,
                 terms=None, normalize=True, loss_weighted=True, eps=1e-16):
        """
        initial_weights : dict of term -> starting weight. Adapted entries are
            overwritten from the second update on (the first only records a
            baseline); the rest are held.
        beta : softmax temperature. Larger sharpens the preference for
            slowly-improving terms; beta = 0 gives uniform weights.
        update_every : epochs per weight update. Losses from every epoch in
            between are averaged, so larger is less noisy but slower to adapt.
        terms : terms to adapt. Default: every term in initial_weights.
        normalize : divide each rate by that term's own previous loss, giving
            a fractional change. Default True.
        loss_weighted : scale each softmax term by its loss share f_i.
            Default True.
        eps : unused; the rate's divisor needs no floor because
            _is_inactive() has already dropped every zero loss.
        """
        super().__init__(initial_weights)
        self.beta = beta
        self.update_every = update_every
        self.normalize = normalize
        self.loss_weighted = loss_weighted
        self.eps = eps
        if terms is None:
            terms = [t for t in LOSS_TERMS if t in self.weights]
        self.terms = list(terms)
        # Previous window mean per term; the rate needs one prior sample, so
        # the first update only records. prev_before holds L_i(t-1) during an
        # update, after prev has advanced to L_i(t).
        self.prev = {}
        self.prev_before = {}
        # Losses banked since the last update, averaged in step 1.
        self._window = {}
        # Per-term trajectory, appended once per update.
        self.history_epochs = []
        self.history = {t: [] for t in self.terms}        # live lambda_i
        self.history_rate = {t: [] for t in self.terms}   # s_i
        self.history_loss = {t: [] for t in self.terms}   # window mean L_i

    @staticmethod
    def _softmax(scores, beta, factors=None):
        """Softmax over a dict of scores, optionally scaled per term.

        scores : dict of term -> score s_i.
        beta : temperature multiplying each score.
        factors : optional term -> multiplier applied to each exponential
            before normalising (the loss shares f_i). Ignored if it would
            zero every entry.

        Returns a dict of term -> weight summing to 1.
        """
        # Subtract the max before exponentiating to keep exp() in range; it
        # cancels between numerator and denominator, so the result is exact.
        top = max(scores.values())
        ex = {t: math.exp(beta * (v - top)) for t, v in scores.items()}
        if factors is not None:
            scaled = {t: ex[t] * factors.get(t, 0.0) for t in ex}
            if sum(scaled.values()) > 0.0:
                ex = scaled
        total = sum(ex.values())
        return {t: v / total for t, v in ex.items()}

    def update(self, epoch, loss_terms, params):
        """Bank this epoch's losses; every update_every epochs, reweight.

        Called once per epoch. Returns True when the weights changed.

        epoch : current epoch, used for the throttle.
        loss_terms : dict of term -> unweighted loss tensor.
        params : unused (no gradients are taken); present to match the
            LossWeights interface.
        """
        # Bank one sample per active term. Each loss is a mean over a freshly
        # resampled collocation batch, so a single draw is too noisy to
        # difference directly.
        for t in self.terms:
            L = loss_terms.get(t)
            if self._is_inactive(L):
                continue
            # detach() then float(): copies the number out of the graph, so
            # the banked history holds plain floats and cannot keep a whole
            # epoch's autograd graph alive in memory.
            self._window.setdefault(t, []).append(float(L.detach()))

        if epoch % self.update_every != 0:
            return False

        # Step 1: average each term's banked samples, then start a new window.
        current = {t: sum(v) / len(v) for t, v in self._window.items() if v}
        self._window = {}
        if not current:
            return False

        # Step 2: change since the previous window mean. A term with no
        # previous sample is skipped and adapts from the next update on.
        rates = {t: current[t] - self.prev[t]
                 for t in current if t in self.prev}
        # Stash L_i(t-1) before prev advances; step 3's divisor needs it.
        self.prev_before = {t: self.prev[t] for t in rates}
        self.prev.update(current)
        if not rates:
            return False

        # Step 3: make it fractional by dividing each rate by its OWN previous
        # loss, so terms whose losses differ by decades become comparable. No
        # epsilon -- _is_inactive() already dropped every zero loss.
        if self.normalize:
            rates = {t: r / abs(self.prev_before[t])
                     for t, r in rates.items()}

        # Step 4: loss shares over the same terms, so an already-tiny term
        # cannot take the budget by fluctuating (s_i is unbounded above).
        factors = None
        if self.loss_weighted:
            tot = sum(current[t] for t in rates)
            if tot > 0.0:
                factors = {t: current[t] / tot for t in rates}

        self.weights.update(self._softmax(rates, self.beta, factors))

        # Step 5: record the trajectory; a term skipped this round stores NaN
        # so a gap in a plotted trace reads as a gap.
        self.history_epochs.append(epoch)
        for t in self.terms:
            self.history[t].append(self.weights[t])
            self.history_rate[t].append(rates.get(t, float("nan")))
            self.history_loss[t].append(current.get(t, float("nan")))
        return True

    def describe(self):
        """One-block summary of the rule and its settings, for the run log."""
        lines = ["Adaptive loss weighting: SoftAdapt "
                 "(Heydari, Thompson & Mehmood 2019)"]
        lines.append("  beta           : {0}   update every {1} epoch(s)".format(
            self.beta, self.update_every))
        lines.append("  rates          : {0}".format(
            "relative, s_i / |L_i(t-1)|" if self.normalize else "raw"))
        lines.append("  loss-weighted  : {0}".format(
            "yes, scaled by L_i / sum_j L_j" if self.loss_weighted else "no"))
        lines.append("  adapting       : {0}".format(", ".join(self.terms)))
        return "\n".join(lines)

    def format_weights(self, terms=None, verbose=False):
        """Summary of the live weights for the training log.

        verbose=True adds, per term, a direction arrow and the rate s_i behind
        the weight. s_i < 0 means the term is still improving; s_i > 0 means
        it is getting worse and is being weighted up in response.
        """
        terms = self.terms if terms is None else terms
        if not verbose:
            return super().format_weights(terms)

        lines = ["    {0:<11s} {1:>10s} {2:>9s}  {3:>12s}".format(
            "term", "weight", "change", "rate s_i")]
        for t in terms:
            g = self.growth(t)
            if g != g:                       # NaN: fewer than two updates
                arrow, gtxt = " ", "    --   "
            else:
                arrow = "^" if g > 1.05 else ("v" if g < 0.95 else "=")
                gtxt = "{0:8.3f}x".format(g)
            r = self.history_rate[t][-1] if self.history_rate[t] else float("nan")
            lines.append("    {0:<11s} {1:10.3e} {2}{3}  {4:12.3e}".format(
                t, self.weights[t], arrow, gtxt, r))
        return "\n".join(lines)

    @staticmethod
    def _is_inactive(L):
        """True for a term carrying no usable signal; see InverseDirichletWeights."""
        # requires_grad is False when a tensor is not connected to any
        # trainable parameter -- the case for a literal zeros() placeholder.
        return (L is None
                or not L.requires_grad
                or float(L.detach()) == 0.0)


def make_weights(kind="fixed", *, initial_weights=None, **kwargs):
    """Build a LossWeights of the requested kind.

    kind : "fixed", "inverse_dirichlet" or "softadapt". Extra keyword
        arguments go to the adaptive rule and are ignored by the fixed one.
    initial_weights : starting weights; DEFAULT_LOSS_WEIGHTS when omitted.
    """
    weights = DEFAULT_LOSS_WEIGHTS if initial_weights is None else initial_weights
    if kind == "fixed":
        return FixedWeights(weights)
    if kind == "inverse_dirichlet":
        return InverseDirichletWeights(initial_weights=weights, **kwargs)
    if kind == "softadapt":
        return SoftAdaptWeights(initial_weights=weights, **kwargs)
    raise ValueError(
        "unknown loss-weight kind {0!r}; expected 'fixed', "
        "'inverse_dirichlet' or 'softadapt'".format(kind))
