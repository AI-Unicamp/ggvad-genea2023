import torch
import math
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

TOKEN_SELF_ATTN_VALUE = -5e4

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k


class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        exact_windowsize = False
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.window_size = window_size
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

    def forward(self, q, k, v, packed_shape, mask = None, input_mask = None):
        mask = default(mask, input_mask)

        pad_value, window_size, causal, look_backward, look_forward, shared_qk = -1, self.window_size, self.causal, self.look_backward, self.look_forward, self.shared_qk

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = dim_head ** -0.5

        if n % window_size != 0:
            print('sequence length must be divisible by window size for local attention', n, window_size)
            assert True

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk) * scale

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        out, *_ = unpack(out, packed_shape, '* n d')
        return out