"""
Nested variable/location graph encoder with a joint forecast, leave-one-out and
reconstruction objective.

score() returns one number per node, the combined Fisher statistic. Channels
are how that number gets built, not something the alerting path should know
about, so channels() is there for diagnostics and nothing downstream calls it.

The hidden state decays between irregular observations. The observation mask
is deliberately not an input: missingness here is flat batteries, and a model
that learns "offline implies anomalous" turns every power cut into an alert.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from strawberrywatch.anomalies import channel_scoring as scoring
from strawberrywatch.models import model_calls

# Graph structure and canonical variables

GRID_MINUTES = 15
GRID_FREQ = f"{GRID_MINUTES}min"

# An observation this close to a grid point is that grid point's target. The
# old rule wanted an exact match, which cost footbridge six times its usable
# supervision because its logger reports at :17 and :28 rather than :15.
TARGET_TOLERANCE = pd.Timedelta(minutes=GRID_MINUTES / 2)

# Separately, and for a different reason: an anchor older than this is not
# trusted enough to predict a delta from. Nothing is dropped at ingestion.
MAX_ANCHOR_STALENESS_STEPS = 4.0

SENTINELS = (-9999.0,)

CANONICAL_VARS = ["conductivity", "depth_dev", "temperature", "do_pct", "float_cond"]
SCORED_VARS = ["conductivity"]

# Prior half life in hours, how long a reading stays informative once it stops
# updating. Conductivity drifts slowly, stage moves fast. These are starting
# points the loss is free to move, not claims.
HALF_LIFE_PRIOR_HOURS = {
    "conductivity": 4.0,
    "depth_dev": 1.0,
    "temperature": 3.0,
    "do_pct": 2.0,
    "float_cond": 4.0,
}

# Clock only. Everything weather-shaped left with the rain handling, and day of
# year went with it: a year-scale seasonal term cannot be estimated from the
# window lengths this model trains on, and it was the channel that made a
# seasonal baseline shift look like a coherent network-wide fault.
CONTEXT_FEATURES = ["hour_sin", "hour_cos"]

COLUMN_MAP = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth_raw",
    "Meter_Hydros21_Temp": "temperature",
    "AtlasSci_DO": "do_pct",
    "AtlasSci_FloatCond": "float_cond",
    "EnviroDIY_Mayfly_Batt": "battery",
}

SITE_INVENTORY = {
    "north_fork_0": ["conductivity", "depth_dev", "temperature"],
    "footbridge": ["conductivity", "depth_dev", "temperature", "do_pct", "float_cond"],
    "south_fork_1": ["conductivity", "depth_dev", "temperature"],
    "south_fork_2": ["conductivity", "depth_dev", "temperature"],
    "oxford": ["conductivity", "depth_dev", "temperature"],
}

SITE_ORDER = ["north_fork_0", "footbridge", "south_fork_1", "south_fork_2", "oxford"]

# Downstream flow, source to destination. The repo contracts the north path to
# a single north_fork_0 to oxford edge; this is the pre-contraction form and
# the directions agree.
FLOW_EDGES = [
    ("north_fork_0", "footbridge"),
    ("footbridge", "oxford"),
    ("south_fork_1", "south_fork_2"),
    ("south_fork_2", "oxford"),
]

# Floor on the predicted scale. Without it the negative log likelihood rewards
# driving the scale to zero on any node the model happens to fit well, which
# blows up the dispersion channel the stuck detector depends on.
MIN_SCALE = 1e-2


def normalize_site_code(value):
    """
    Collapse the ways one site writes its own name. north_fork_0's archive has
    "North Fork #0" for the first 15,825 rows and "north_fork_0" after, which is
    a rename mid deployment, not two sites. Stripping to alphanumerics makes
    both "northfork0" and lets the multi site guard stay strict about the case
    it is there for.
    """
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@dataclass(frozen=True)
class NodeSpec:
    site: str
    var: str

    @property
    def key(self):
        return f"{self.site}.{self.var}"


def build_node_registry(inventory=SITE_INVENTORY, site_order=SITE_ORDER):
    """One node per (site, variable). A site missing a sensor has fewer nodes."""
    nodes, node_site, node_var = [], [], []
    for s_idx, site in enumerate(site_order):
        for var in inventory.get(site, []):
            nodes.append(NodeSpec(site, var))
            node_site.append(s_idx)
            node_var.append(CANONICAL_VARS.index(var))
    return (
        nodes,
        torch.tensor(node_site, dtype=torch.long),
        torch.tensor(node_var, dtype=torch.long),
    )


def build_site_matrix(node_site, num_sites):
    n = len(node_site)
    m = torch.zeros(num_sites, n)
    m[node_site, torch.arange(n)] = 1.0
    return m


def build_variable_adjacency(node_site):
    """
    Level one. A node talks to the other sensors at its own site and nowhere
    else. This is where two conductivity sensors disagreeing becomes a hardware
    fault signal distinct from a spill, since both moving together is an event
    and one moving alone is a failure.
    """
    a = (node_site.unsqueeze(0) == node_site.unsqueeze(1)).float()
    a.fill_diagonal_(0.0)
    return a


def junction_structure(site_order=SITE_ORDER, edges=FLOW_EDGES):
    """
    Group direct upstream contributors by their destination. A destination with
    one contributor takes all of its flow from it. A destination with several is
    a junction, and the split between them is learned.
    """
    idx = {s: i for i, s in enumerate(site_order)}
    upstream = {}
    for src, dst in edges:
        upstream.setdefault(idx[dst], []).append(idx[src])
    return upstream


def inventory_matrix(sites=SITE_ORDER, inventory=SITE_INVENTORY):
    m = torch.zeros(len(sites), len(CANONICAL_VARS))
    for i, s in enumerate(sites):
        for v in inventory[s]:
            m[i, CANONICAL_VARS.index(v)] = 1.0
    return m


def gaussian_nll(pred, scale, target, mask):
    """
    Negative log likelihood in place of squared error, masked to the entries
    where it can honestly be computed.

    The log scale term is what stops the model from claiming certainty
    everywhere, and it is what makes the dispersion channel mean anything.
    """
    if not bool(mask.any()):
        return pred.sum() * 0.0
    z = (target - pred) / scale
    return (0.5 * z**2 + torch.log(scale))[mask].mean()


# Encoder


def masked_message(weights, messages):
    """Weighted average over live senders, zero where a node has none."""
    denom = weights.sum(-1, keepdim=True)
    out = torch.bmm(weights, messages) / denom.clamp(min=1e-6)
    return torch.where(denom > 0, out, torch.zeros_like(out))


class DecayLSTMCell(nn.Module):
    """
    LSTM whose state and input decay with elapsed time between observations,
    since the rest of the system is LSTM shaped. Both decays are driven only by
    elapsed time.

    Parameterised as log half life rather than a raw rate. The previous form
    used exp(-clamp(delta * w.abs())) initialised at w = 0, where the absolute
    value has no gradient, so nothing ever moved it off zero and the decay was
    dead rather than merely inert. Half life is always positive and
    differentiable everywhere, and it initialises to a stated prior.

    The observation mask is deliberately not an input. Feeding it in exploits
    informative missingness, which is right where a missing measurement is a
    clinical decision. Here missingness is flat batteries and network outages,
    and a model that learns "offline implies anomalous" turns every power
    failure into an alert.
    """

    def __init__(self, input_dim, hidden_dim, prior_hours):
        super().__init__()
        self.cell = nn.LSTMCell(input_dim, hidden_dim)
        steps = torch.tensor([h * 60.0 / GRID_MINUTES for h in prior_hours])
        self.log_hl_x = nn.Parameter(torch.log(steps))
        self.log_hl_h = nn.Parameter(torch.full((hidden_dim,), math.log(8.0)))

    def half_life_x(self):
        return torch.exp(self.log_hl_x)

    def input_decay(self, delta, node_var):
        return torch.pow(0.5, delta / self.half_life_x()[node_var])

    def forward(self, x, delta, state):
        h, c = state
        gamma = torch.pow(0.5, delta.unsqueeze(-1) / torch.exp(self.log_hl_h))
        return self.cell(x, (h * gamma, c))


class TailUp(nn.Module):
    """
    Directed downstream transport over the site graph. Forward only.

    The reverse path from the previous pass is deleted. It was there because an
    upstream site and its downstream neighbour share a storm and a baseflow
    regime, so a downstream reading is informative about an upstream one even
    though water does not flow that way. With weather out of the model that
    argument still holds for baseflow, but the measurement settled it: detection
    was bit-identical with theta_rev zeroed, so the reverse path was carrying
    nothing the forward path and the sibling graph did not already carry.

    What stays: the junction split is learned rather than assumed. Tail-up
    weighting wants a discharge ratio, nothing in the data measures discharge,
    and a hardcoded 0.5 is an assumption the model has to work around. A softmax
    over each junction's contributors makes it something the data fits, and the
    learned value becomes a physical claim a field survey can check later.

    Flow isolation is the property to check on the trained weights: the forward
    matrix is nonzero only along downstream paths, so on a tree it never
    connects the north fork to the south fork at any number of hops, and cross
    fork response stays exactly zero rather than merely small.
    """

    def __init__(self, hidden, num_sites, upstream, k_hops=2):
        super().__init__()
        self.num_sites = num_sites
        self.upstream = upstream
        self.k_hops = k_hops
        self.logits = nn.ParameterDict(
            {
                str(dst): nn.Parameter(torch.zeros(len(src)))
                for dst, src in upstream.items()
                if len(src) > 1
            }
        )
        # Not zeros. At zero the hop weight kills the gradient to msg entirely,
        # which is the dead-parameter-at-init failure the old decay rate already
        # cost us once. Uniform over hops starts as a plain average and is free
        # to move.
        self.theta = nn.Parameter(torch.full((k_hops,), 1.0 / k_hops))
        self.msg = nn.Linear(hidden, hidden)
        # Follows the module through .double()/.to(), so direct_matrix can
        # allocate in the right dtype even for a topology with no junctions and
        # therefore no parameters to read it from.
        self.register_buffer("dtype_probe", torch.zeros(0))

    def direct_matrix(self, device, dtype=None):
        """W[src, dst] is the share of dst's inflow contributed by src."""
        dtype = dtype or self.dtype_probe.dtype
        rows, cols, vals = [], [], []
        for dst, srcs in self.upstream.items():
            if len(srcs) == 1:
                w = torch.ones(1, device=device, dtype=dtype)
            else:
                w = torch.softmax(self.logits[str(dst)].to(device=device, dtype=dtype), dim=0)
            for j, src in enumerate(srcs):
                rows.append(src)
                cols.append(dst)
                vals.append(w[j])
        base = torch.zeros(self.num_sites, self.num_sites, device=device, dtype=dtype)
        return base.index_put(
            (torch.tensor(rows, device=device), torch.tensor(cols, device=device)),
            torch.stack(vals),
        )

    def transition(self, device, dtype=None):
        """
        Receive matrix: row dst, column src. The transpose of W, whose rows are
        already the junction softmax and so already sum to one.
        """
        return self.direct_matrix(device, dtype).t()

    def reach_matrix(self, device, dtype=None):
        """Sum of the hop powers actually used, for the isolation report."""
        p = self.transition(device, dtype)
        acc, power = p, p
        for _ in range(self.k_hops - 1):
            power = power @ p
            acc = acc + power
        return acc

    def forward(self, site_h, site_live):
        dev, dt = site_h.device, site_h.dtype
        p = self.transition(dev, dt)
        live = site_live.to(dt).unsqueeze(1)
        m = self.msg(site_h)
        acc = torch.zeros_like(site_h)
        power = p
        for k in range(self.k_hops):
            acc = acc + self.theta[k] * masked_message(power.unsqueeze(0) * live, m)
            if k + 1 < self.k_hops:
                power = power @ p
        return acc


class EncoderOut(NamedTuple):
    """
    h         (B, N, hidden)  node state after site_mix
    site_ctx  (B, N, hidden)  transported site context, broadcast back to nodes
    anchor    (B, N)          last value, already zeroed at masked nodes
    live      (B, N)          bool, anchor fresh enough to predict a delta from
    """

    h: torch.Tensor
    site_ctx: torch.Tensor
    anchor: torch.Tensor
    live: torch.Tensor


class NestedEncoder(nn.Module):
    """
    The shared trunk. All three models are this encoder plus different heads,
    which is the whole point: if the trunks differ the comparison measures
    trunks rather than heads.

    Order of operations, per timestep:
      value decayed by the per-variable half life, then masked if the node is
      hidden; concatenated with log staleness, context, and the variable, site
      and node embeddings; projected; multiplied by a static gate built from the
      site inventory and the site embedding, EA-LSTM style; through the decay
      LSTM; LayerNorm; sibling exchange over a_var.
    Then once, at the end: pool to sites, tail-up transport, site_mix.

    Sibling messaging runs at every step rather than once at the end. It costs
    about 1.4x per epoch, it does not improve mean error, and it was the more
    reliable configuration on every seed, which is why it is the only one here.

    LayerNorm on the recurrent output is purely about training dynamics: this
    architecture reaches a loss basin roughly a third better than the site-node
    baseline, but at a shared learning rate it gets there by luck, with
    transient excursions after it has already descended.

    node_mask is (B, N) bool, True where the node's own reading is hidden. For a
    hidden node BOTH the value channel and the staleness channel are zeroed at
    every timestep, and the anchor is zeroed too. Embeddings and context stay,
    so a masked node still carries which sensor at which site it is and what time
    it is; what it loses is every path to its own readings. Freshness gating of
    the sibling and site graphs still reads the raw staleness, which is timing
    rather than measurement: it gates who talks, not what is said, and the
    leak assertion below holds because of it, not in spite of it.
    """

    def __init__(
        self,
        num_sites,
        num_vars,
        num_context,
        num_nodes,
        hidden=64,
        var_emb=8,
        site_emb=12,
        k_hops=2,
        upstream=None,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_nodes = num_nodes

        self.var_emb = nn.Embedding(num_vars, var_emb)
        self.site_emb = nn.Embedding(num_sites, site_emb)
        self.node_emb = nn.Embedding(num_nodes, site_emb)

        in_dim = 1 + 1 + num_context + var_emb + site_emb * 2
        self.in_proj = nn.Linear(in_dim, hidden)
        self.static_gate = nn.Sequential(nn.Linear(num_vars + site_emb, hidden), nn.Sigmoid())

        prior = [HALF_LIFE_PRIOR_HOURS[v] for v in CANONICAL_VARS]
        self.rnn = DecayLSTMCell(hidden, hidden, prior)
        self.norm = nn.LayerNorm(hidden)

        self.sib_msg = nn.Linear(hidden, hidden)
        self.sib_mix = nn.Linear(hidden * 2, hidden)
        self.tail_up = TailUp(
            hidden, num_sites, upstream if upstream is not None else junction_structure(), k_hops
        )
        self.site_mix = nn.Linear(hidden * 2, hidden)

    def _siblings(self, h, live, a_var):
        sib = masked_message(a_var.unsqueeze(0) * live.unsqueeze(1), self.sib_msg(h))
        return self.sib_mix(torch.cat([h, sib], dim=-1))

    def forward(
        self,
        values,
        staleness,
        context,
        node_site,
        node_var,
        site_inventory,
        a_var,
        site_matrix,
        node_mask=None,
    ):
        b, t, n = values.shape
        dev, dt = values.device, values.dtype

        keep = None if node_mask is None else (~node_mask).to(dt)

        v_emb = self.var_emb(node_var)
        s_emb = self.site_emb(node_site)
        n_emb = self.node_emb(torch.arange(n, device=dev))
        gate = self.static_gate(torch.cat([site_inventory[node_site], s_emb], dim=-1))

        h = torch.zeros(b * n, self.hidden, device=dev, dtype=dt)
        c = torch.zeros(b * n, self.hidden, device=dev, dtype=dt)

        for step in range(t):
            d = staleness[:, step]
            # Post scaling the node mean is zero, so a stale reading decays
            # toward zero and there is no separate mean to carry around.
            own_val = values[:, step] * self.rnn.input_decay(d, node_var)
            own_stale = torch.log1p(d)
            if keep is not None:
                own_val = own_val * keep
                own_stale = own_stale * keep
            x = torch.cat(
                [
                    own_val.unsqueeze(-1),
                    own_stale.unsqueeze(-1),
                    context[:, step, :].unsqueeze(1).expand(b, n, -1),
                    v_emb.unsqueeze(0).expand(b, -1, -1),
                    s_emb.unsqueeze(0).expand(b, -1, -1),
                    n_emb.unsqueeze(0).expand(b, -1, -1),
                ],
                dim=-1,
            )
            x = self.in_proj(x) * gate.unsqueeze(0)
            h, c = self.rnn(x.reshape(b * n, -1), d.reshape(b * n), (h, c))

            # A sensor that has gone quiet stops talking to its siblings rather
            # than broadcasting its decayed value into their state every step.
            live_t = (d <= MAX_ANCHOR_STALENESS_STEPS).to(dt)
            h = self._siblings(self.norm(h).view(b, n, -1), live_t, a_var).reshape(b * n, -1)

        h = self.norm(h).view(b, n, -1)
        live = (staleness[:, -1] <= MAX_ANCHOR_STALENESS_STEPS).to(dt)

        w_site = site_matrix.unsqueeze(0) * live.unsqueeze(1)
        s_denom = w_site.sum(-1, keepdim=True)
        site_h = torch.bmm(w_site, h) / s_denom.clamp(min=1.0)
        site_live = s_denom.squeeze(-1) > 0

        transport = self.tail_up(site_h, site_live)
        idx = node_site.view(1, n, 1).expand(b, n, transport.shape[-1])
        site_ctx = transport.gather(1, idx)

        mixed = self.site_mix(torch.cat([h, site_ctx], dim=-1))

        # The anchor has to be masked too, or a hidden node gets its own last
        # reading back through the residual connection and the leave-one-out
        # prediction rides the value it is meant to reconstruct. A masked node
        # anchors at 0, the post-scaling mean.
        anchor = values[:, -1] if keep is None else values[:, -1] * keep
        return EncoderOut(mixed, site_ctx, anchor, live.bool())


# Heads

ENCODER_KEYS = (
    "values",
    "staleness",
    "context",
    "node_site",
    "node_var",
    "site_inventory",
    "a_var",
    "site_matrix",
)

# Heads are drawn after the trunk so that models sharing a seed have
# byte-identical encoder initialisations. Without this the trunks would differ
# from step zero and any comparison would be measuring that instead.
HEAD_SEED_OFFSET = 10_000


class ScaleHead(nn.Module):
    """
    Location and scale instead of a point.

    Used by every head so the channels are comparable: each one reports a
    standardised residual against a scale the model itself predicted, rather
    than a raw magnitude in whatever units that head happens to work in.
    """

    def __init__(self, in_dim, out_dim=1, min_scale=MIN_SCALE):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.log_scale = nn.Linear(in_dim, out_dim)
        self.min_scale = min_scale
        self.out_dim = out_dim

    def forward(self, h):
        mu = self.mu(h)
        scale = F.softplus(self.log_scale(h)) + self.min_scale
        if self.out_dim == 1:
            return mu.squeeze(-1), scale.squeeze(-1)
        return mu, scale


class ReconHead(nn.Module):
    """
    Decodes the whole input window for one node from its final state, as a
    per-step mu and scale.

    The bottleneck is 8 against hidden 64 and it has to stay tight: widen it and
    reconstruction degenerates into an identity map, at which point the channel
    measures nothing and scores beautifully while doing it. The asymmetry
    follows ML4ITS/mtad-gat-pytorch, where the forecast head is the deeper one
    and reconstruction is shallow.
    """

    def __init__(self, hidden, window, bottleneck=8, min_scale=MIN_SCALE):
        super().__init__()
        self.window = window
        self.bottleneck = nn.Linear(hidden, bottleneck)
        self.out = nn.Linear(bottleneck, window * 2)
        self.min_scale = min_scale

    def forward(self, h):
        b, n, _ = h.shape
        z = self.out(torch.tanh(self.bottleneck(h))).view(b, n, self.window, 2)
        mu = z[..., 0]
        scale = F.softplus(z[..., 1]) + self.min_scale
        return mu, scale


class _Nested(nn.Module):
    """Shared trunk, shared plumbing, no head-specific behaviour."""

    channel_names: tuple[str, ...] = ()

    def __init__(
        self,
        num_sites,
        num_vars,
        num_context,
        num_nodes,
        seed,
        hidden=64,
        var_emb=8,
        site_emb=12,
        window=24,
    ):
        super().__init__()
        self.hidden = hidden
        self.window = window
        self.seed = seed
        torch.manual_seed(seed)
        self.encoder = NestedEncoder(
            num_sites,
            num_vars,
            num_context,
            num_nodes,
            hidden=hidden,
            var_emb=var_emb,
            site_emb=site_emb,
        )
        torch.manual_seed(seed + HEAD_SEED_OFFSET)
        self.build_heads(hidden)

    def build_heads(self, hidden):
        raise NotImplementedError

    # Encoding and forecasting

    def encode(self, batch, node_mask=None):
        out = self.encoder(**{k: batch[k] for k in ENCODER_KEYS}, node_mask=node_mask)
        return torch.cat([out.h, out.site_ctx], dim=-1), out.anchor

    def forecast(self, h, anchor):
        delta, scale = self.head(h)
        return anchor + delta, scale

    def loo_forecast(self, h, anchor):
        """
        ModelA has no leave-one-out head, so it falls back to its forecast head.
        Nothing scores that path, since A has no leave-one-out channel, but it lets the
        masked-input leak assertion run against all three models, which is what
        the assertion is for: the mask lives in the shared encoder, so a leak
        would be a leak for every model that uses it.
        """
        delta, scale = getattr(self, "head_loo", self.head)(h)
        return anchor + delta, scale

    @staticmethod
    def anchor_fresh(batch):
        """Recomputed rather than returned by encode, so encode stays a pair."""
        return batch["staleness"][:, -1] <= MAX_ANCHOR_STALENESS_STEPS

    def forecast_terms(self, batch):
        h, anchor = self.encode(batch)
        mu, scale = self.forecast(h, anchor)
        mask = batch["target_mask"] & self.anchor_fresh(batch)
        return h, mu, scale, mask

    # Leave-one-out

    def sample_node_mask(self, batch, p, generator=None):
        v = batch["values"]
        b, _t, n = v.shape
        return torch.rand(b, n, device=v.device, generator=generator) < p

    @torch.no_grad()
    def predict_all_masked(self, batch):
        """
        Every node's leave-one-out prediction in one pass.

        Batches N copies of a single window, copy i hiding node i, so this costs
        one forward at batch N rather than N forwards at batch 1. Ranking a
        faulted node needs the masked prediction for every node, so the naive
        version would dominate the whole sweep.
        """
        was_training = self.training
        self.eval()
        v = batch["values"]
        assert v.shape[0] == 1, "predict_all_masked expects a single window"
        n = v.shape[2]
        rep = {
            k: (x.expand(n, *x.shape[1:]) if torch.is_tensor(x) and x.dim() >= 3 else x)
            for k, x in batch.items()
        }
        mask = torch.eye(n, device=v.device, dtype=torch.bool)
        h, anchor = self.encode(rep, node_mask=mask)
        pred, scale = self.loo_forecast(h, anchor)
        if was_training:
            self.train()
        d = torch.arange(n, device=v.device)
        return pred[d, d], scale[d, d]

    def masked_predictor(self, batch, node_index):
        """
        A closure for common.masked_input_leak: values in, that window's masked
        predictions out. Uses the single-node mask rather than the batched one,
        so the assertion tests exactly the path the LOO channel uses per node.
        """

        def predict(values):
            b, _t, n = values.shape
            mask = torch.zeros(b, n, dtype=torch.bool, device=values.device)
            mask[:, node_index] = True
            h, anchor = self.encode({**batch, "values": values}, node_mask=mask)
            pred, _scale = self.loo_forecast(h, anchor)
            return pred[0]

        return predict

    # Scoring

    def base_channels(self, batch):
        """Forecast pass plus the free dispersion channel, shared by all three."""
        h, mu, scale, mask = self.forecast_terms(batch)
        chans = {
            "excursion": scoring.excursion(
                mu, scale, batch["target"], batch["target_mask"], self.anchor_fresh(batch)
            ),
            "dispersion": scoring.dispersion(scale, batch["values"], obs_mask=batch["obs_mask"]),
        }
        return h, mu, scale, mask, chans

    def channels(self, batch):
        raise NotImplementedError


class _ForecastLooBase(_Nested):
    """
    Forecast plus leave-one-out, on two separate passes.

      pass 1: clean forward, no node masking, forecast NLL
      pass 2: masked forward with node dropout p=0.15, leave-one-out NLL
      total = forecast_nll + lambda * loo_nll

    The separation is not cosmetic. Applying node dropout to a single shared
    pass moved 68/95 nominal interval coverage from 71/94 to 90/100, because the
    model hedged its forecast scale against inputs that might be masked. Two
    passes and two scale heads keep the forecast scale answering "how wrong is
    my next-step forecast" and the LOO scale answering "how wrong am I about a
    node I cannot see", which are different questions with different answers.

    Channels: excursion, loo, dispersion.
    """

    channel_names = ("excursion", "loo", "dispersion")

    def __init__(self, *args, lam_loo=0.5, node_dropout=0.15, **kw):
        super().__init__(*args, **kw)
        self.lam_loo = lam_loo
        self.node_dropout = node_dropout

    def build_heads(self, hidden):
        self.head = ScaleHead(hidden * 2)
        self.head_loo = ScaleHead(hidden * 2)

    def loo_term(self, batch, generator=None):
        mask = self.sample_node_mask(batch, self.node_dropout, generator)
        h, anchor = self.encode(batch, node_mask=mask)
        pred, scale = self.loo_forecast(h, anchor)
        # Score only where a node was actually hidden. Scoring the visible ones
        # would just be the forecast objective again under a second head.
        hidden = batch["target_mask"] & mask
        return gaussian_nll(pred, scale, batch["target"], hidden)

    def loss(self, batch, generator=None):
        _h, mu, scale, mask = self.forecast_terms(batch)
        forecast = gaussian_nll(mu, scale, batch["target"], mask)
        loo = self.loo_term(batch, generator)
        total = forecast + self.lam_loo * loo
        return total, {"forecast": float(forecast), "loo": float(loo)}

    @torch.no_grad()
    def channels(self, batch):
        was_training = self.training
        self.eval()
        _h, _mu, _scale, _mask, chans = self.base_channels(batch)
        loo_pred, loo_scale = self.predict_all_masked(batch)
        chans["loo"] = scoring.loo(
            loo_pred.unsqueeze(0), loo_scale.unsqueeze(0), batch["target"], batch["target_mask"]
        )
        if was_training:
            self.train()
        return chans


class CobbleShoal(_ForecastLooBase):
    """
    Everything in ModelB plus a reconstruction head.

    loss = forecast_nll + lambda_loo * loo_nll + lambda_rec * rec_nll, with the
    reconstruction NLL masked to observed steps only.

    The rationale, stated so it can fail: the forecast channel is anchored on
    the last value, the leave-one-out channel is anchored on the neighbours, and
    reconstruction is anchored on neither, so it is the only channel that can
    see a window whose shape is wrong when the level is right and the network
    agrees. If it does not beat A and B on decouple and slow_all, the head is
    parameters for nothing.

    Reconstruction is computed from the clean pass, so C costs two forwards per
    step and not three; the measured cost is reported rather than assumed.

    Channels: excursion, loo, reconstruction, dispersion.
    """

    INPUT_CONTRACT = model_calls.NESTED_NODE_BATCH

    # Nothing. Rain is handled outside this model by anomalies/rain_gate.py,
    # which moves the threshold and never the score. Empty rather than omitted:
    # an empty tuple says this was checked.
    BUILTIN_SUPPORT = ()

    channel_names = ("excursion", "loo", "reconstruction", "dispersion")

    @classmethod
    def from_metadata(cls, metadata):
        """
        Rebuild from a trained metadata blob. Sizes come from the creek
        structure this model defines, not from the caller, because the node
        count is a property of the site inventory rather than of the window.
        """
        return cls(
            len(SITE_ORDER),
            len(CANONICAL_VARS),
            len(CONTEXT_FEATURES),
            len(build_node_registry()[0]),
            int(metadata.get("seed", 0)),
            window=int(metadata.get("window", 24)),
        )

    def __init__(self, *args, lam_rec=0.5, **kw):
        super().__init__(*args, **kw)
        self.lam_rec = lam_rec

    def build_heads(self, hidden):
        super().build_heads(hidden)
        self.head_rec = ReconHead(hidden, self.window)

    def reconstruct(self, h):
        """The bottleneck reads the node state only, the leading hidden columns."""
        return self.head_rec(h[..., : self.hidden])

    def rec_term(self, batch, h):
        x_hat, rec_scale = self.reconstruct(h)
        target = batch["values"].transpose(1, 2)
        mask = batch["obs_mask"].transpose(1, 2)
        return gaussian_nll(x_hat, rec_scale, target, mask)

    def loss(self, batch, generator=None):
        h, mu, scale, mask = self.forecast_terms(batch)
        forecast = gaussian_nll(mu, scale, batch["target"], mask)
        rec = self.rec_term(batch, h)
        loo = self.loo_term(batch, generator)
        total = forecast + self.lam_loo * loo + self.lam_rec * rec
        return total, {"forecast": float(forecast), "loo": float(loo), "reconstruction": float(rec)}

    @torch.no_grad()
    def channels(self, batch):
        was_training = self.training
        self.eval()
        h, _mu, _scale, _mask, chans = self.base_channels(batch)
        loo_pred, loo_scale = self.predict_all_masked(batch)
        chans["loo"] = scoring.loo(
            loo_pred.unsqueeze(0), loo_scale.unsqueeze(0), batch["target"], batch["target_mask"]
        )
        x_hat, rec_scale = self.reconstruct(h)
        chans["reconstruction"] = scoring.reconstruction(
            x_hat, rec_scale, batch["values"], batch["obs_mask"]
        )
        if was_training:
            self.train()
        return chans

    # Detector-facing interface

    @torch.no_grad()
    def score(self, batch, nulls, return_channels=False):
        """
        One number per node: the combined Fisher statistic. Larger is more
        anomalous. This is the whole contract the alerting path needs.

        The detector deliberately cannot see a channel from here. Channels are
        an implementation detail of how the score is built, and every caller
        that learned their names became a place the model could not change.
        Pass return_channels=True to get the raw per-channel scores alongside,
        for diagnostics that genuinely want them.

        nulls is a ChannelNulls fitted at calibration time and loaded from the
        saved artifacts. Nothing is fitted here: this runs on the serving path.
        """
        chans = self.channels(batch)
        combined = scoring.combine(pvalues=nulls.pvalues(chans), rule="fisher")
        return (combined, chans) if return_channels else combined


def build_cobble_shoal(num_sites, num_vars, num_context, num_nodes, seed, window):
    """
    Construct with the same argument order the comparison harness used, so a
    checkpoint from that harness loads into this class without a shim.
    """
    return CobbleShoal(num_sites, num_vars, num_context, num_nodes, seed, window=window)
