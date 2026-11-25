import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=None,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        # default to input_dim so the 1x1 conv stays a mixing layer instead of a
        # hidden bottleneck when the action dim differs from the old default of 10
        smoothed_dim = smoothed_dim or input_dim
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x

class IDM(nn.Module):

   def __init__(
       self,
       z_dim,
       code_dim=3,
       hidden=512,
       code_bn=True,
       out_act=None,
       input_mode="difference",
   ):
       super().__init__()
       if input_mode not in ("difference", "concatenation"):
           raise ValueError(
               "input_mode must be difference|concatenation, "
               f"got {input_mode!r}"
           )
       self.z_dim = z_dim
       self.code_dim = code_dim
       self.input_mode = input_mode

       in_dim = z_dim if input_mode == "difference" else 2 * z_dim
       self.net = nn.Sequential(
           nn.Linear(in_dim, hidden),
           nn.GELU(),
           nn.Linear(hidden, hidden),
           nn.GELU(),
           nn.Linear(hidden, code_dim),
       )
       self.code_bn = nn.BatchNorm1d(code_dim, affine=False) if code_bn else None
       self.out_norm = {None: nn.Identity(), "tanh": nn.Tanh()}[out_act]


   def _apply_code_bn(self, h):
       if self.code_bn is None:
           return h
       shape = h.shape
       h = self.code_bn(h.reshape(-1, self.code_dim))
       return h.reshape(shape)


   def forward(self, z_t, z_tp1):
       if self.input_mode == "difference":
           x = z_tp1 - z_t
       else:
           x = torch.cat((z_t, z_tp1), dim=-1)
       h = self._apply_code_bn(self.net(x))
       return self.out_norm(h)

_SPLITMIX_ADD = 0x9E3779B97F4A7C15 - (1 << 64)
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_SPLITMIX_MUL2 = 0x94D049BB133111EB - (1 << 64)


def _logical_rshift(x, k):
    """Right shift without sign extension. torch's >> on int64 is arithmetic,
    which would smear the sign bit through the mixer and bias the output."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _splitmix64(x):
    z = x + _SPLITMIX_ADD
    z = (z ^ _logical_rshift(z, 30)) * _SPLITMIX_MUL1
    z = (z ^ _logical_rshift(z, 27)) * _SPLITMIX_MUL2
    return z ^ _logical_rshift(z, 31)


_HASH_BITS = 53
_HASH_RANGE = 1 << _HASH_BITS


def transition_hash(episode_idx, step_idx, label_seed=0):
    key = episode_idx.to(torch.int64) * 1_000_003 + step_idx.to(torch.int64)
    key = key + int(label_seed) * 2_654_435_761  # odd; separates seeds pre-mixing
    return _splitmix64(key) & (_HASH_RANGE - 1)


def make_mask(episode_idx, step_idx, frac, label_seed=0, valid=None):
    frac = float(frac)  # a stray string override would otherwise fail obscurely
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"label_frac must be in [0, 1], got {frac}")

    if frac >= 1.0:
        keep = torch.ones_like(episode_idx, dtype=torch.bool)
    elif frac <= 0.0:
        keep = torch.zeros_like(episode_idx, dtype=torch.bool)
    else:
        # integer threshold: hash is uniform over [0, 2**53), so the keep rate is
        # exactly threshold / 2**53, with no float conversion on the device
        threshold = int(frac * _HASH_RANGE)
        keep = transition_hash(episode_idx, step_idx, label_seed) < threshold

    if valid is not None:
        keep = keep & valid
    return keep


def make_grouped_action_mask(
    episode_idx, step_idx, group_size, frac, label_seed=0, valid=None
):
    """Return fixed raw-action labels for overlapping blocks.

    Consecutive, episode-local groups share one label decision. ``step_idx``
    identifies each block's first raw action, so the result has one extra
    dimension of length ``group_size`` for its constituent raw actions.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    offsets = torch.arange(group_size, device=step_idx.device)
    raw_step_idx = step_idx.unsqueeze(-1) + offsets
    group_start = raw_step_idx - torch.remainder(raw_step_idx, group_size)
    group_episode_idx = episode_idx.unsqueeze(-1).expand_as(group_start)
    return make_mask(
        group_episode_idx,
        group_start,
        frac,
        label_seed=label_seed,
        valid=valid,
    )


def make_episode_action_mask(
    episode_idx, group_size, frac, label_seed=0, valid=None
):
    """Label all raw actions from a deterministic fraction of full episodes."""
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    episode_keep = make_mask(
        episode_idx,
        torch.zeros_like(episode_idx),
        frac,
        label_seed=label_seed,
    )
    keep = episode_keep.unsqueeze(-1).expand(*episode_keep.shape, group_size)
    if valid is not None:
        keep = keep & valid
    return keep