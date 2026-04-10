import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from haversine import haversine_vector, Unit


# --------------------------------------------------------------------------- #
# 1. LocalPerformerAttention with 2-layer Temporal Convolution
# --------------------------------------------------------------------------- #
class LocalPerformerAttention(nn.Module):
    def __init__(self, dim, heads=4, bias_heads=4, window_size=8, stride=None,
                 coordinates=None, device='cpu', sigma_init=6,
                 kappa_init=0.5, tau_init=12.0, num_samples=3, lambda_u_init=0.1):
        super().__init__()
        assert dim % heads == 0 and 0 < bias_heads <= heads

        self.dim, self.heads, self.bias_heads = dim, heads, bias_heads
        self.dim_head = dim // heads
        self.window, self.stride = window_size, stride or window_size
        self.num_samples = num_samples

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        if coordinates is None:
            raise ValueError("station coordinates required")
        coords = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
        self.register_buffer("dist_mat", self._pairwise_dist(coords))
        self.register_buffer("bear_mat", self._pairwise_bearing(coords))

        # 预计算连线均匀采样权重
        self.register_buffer("adv_weights", self._precompute_sampling_weights(coords, num_samples))

        self.N = self.dist_mat.size(0)

        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.lambda_u = nn.Parameter(torch.tensor(float(lambda_u_init)))

        self.station_bias = nn.Parameter(torch.zeros(self.N))
        self.eps = 1e-6

    def _precompute_sampling_weights(self, coords, M):
        N = coords.shape[0]
        alphas = torch.linspace(0, 1, M + 2)[1:-1].to(coords.device)
        s_nodes, e_nodes = coords.reshape(N, 1, 1, 2), coords.reshape(1, N, 1, 2)
        pts = s_nodes + alphas.reshape(1, 1, M, 1) * (e_nodes - s_nodes)
        dists = torch.norm(pts.reshape(N * N * M, 1, 2) - coords.reshape(1, N, 2), dim=-1).reshape(N, N, M, N)
        w = 1.0 / (dists + 1e-6).pow(2)
        return (w / w.sum(dim=-1, keepdim=True)).mean(dim=2)

    def forward(self, x, u10, v10):
        B, L, N, C = x.shape
        device = x.device
        out = torch.zeros_like(x)

        ws = torch.sqrt(u10 ** 2 + v10 ** 2 + 1e-8)
        flow_rad = torch.atan2(v10, u10)
        cos_f, sin_f = torch.cos(flow_rad), torch.sin(flow_rad)

        sigma = F.softplus(self.sigma) + self.eps
        G = torch.exp(- self.dist_mat.pow(2) / (2 * sigma ** 2))

        for s in range(0, L, self.stride):
            e = min(s + self.window, L)
            wlen = e - s
            blk = x[:, s:e]
            bsN = B * N

            qkv = self.to_qkv(blk.permute(0, 2, 1, 3).contiguous().reshape(bsN, wlen, C))
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(bsN, wlen, self.heads, self.dim_head).transpose(1, 2)
            k = k.reshape(bsN, wlen, self.heads, self.dim_head).transpose(1, 2)
            v = v.reshape(bsN, wlen, self.heads, self.dim_head).transpose(1, 2)
            qφ = F.elu(q) + 1.
            kφ = F.elu(k) + 1.

            ws_blk = ws[:, s:e]
            cf_blk = cos_f[:, s:e][..., None]
            sf_blk = sin_f[:, s:e][..., None]
            cosβ = (torch.cos(self.bear_mat)[None, None] * cf_blk +
                    torch.sin(self.bear_mat)[None, None] * sf_blk)
            wind = torch.tanh(self.kappa * ws_blk[..., None] * cosβ)

            bias_node = torch.einsum('ij,btij->bti', G, 1. + wind)
            bias_node = bias_node + self.station_bias[None, None, :]
            bias_bn = bias_node.permute(0, 2, 1).contiguous().reshape(bsN, 1, wlen, 1)

            dt = torch.arange(wlen, device=device, dtype=x.dtype)
            decay_t = torch.exp(-dt / (F.softplus(self.tau) + self.eps))[None, None, :, None]

            kφ = kφ * (1. + bias_bn) * decay_t

            KV = torch.einsum('b h l d, b h l e -> b h d e', kφ, v)
            Ksum = kφ.sum(dim=2)
            num = torch.einsum('b h t d, b h d e -> b h t e', qφ, KV)
            den = torch.einsum('b h t d, b h d -> b h t', qφ, Ksum).unsqueeze(-1)
            out_blk = num / (den + self.eps)

            out_blk = out_blk.transpose(1, 2).contiguous().reshape(bsN, wlen, C)
            out_blk = self.to_out(out_blk).reshape(B, N, wlen, C).permute(0, 2, 1, 3)
            out[:, s:e] = out_blk

        # 基于平流的传输更新 (Residual Advection)
        u_p = torch.einsum('btk, ijk -> btij', u10, self.adv_weights)
        v_p = torch.einsum('btk, ijk -> btij', v10, self.adv_weights)
        s_ij = torch.clamp(u_p * torch.sin(self.bear_mat.T) + v_p * torch.cos(self.bear_mat.T), min=0)
        adv_w = s_ij / (self.dist_mat + 1e-3)
        adv_term = torch.einsum('btij, btjc -> btic', adv_w, out) - out * adv_w.sum(dim=-1, keepdim=True)

        out = out + F.softplus(self.lambda_u) * adv_term

        return out

    def _pairwise_dist(self, xy):
        coords = xy.cpu().numpy()
        dist = haversine_vector(coords, coords, Unit.KILOMETERS, comb=True)
        N = coords.shape[0]
        return torch.tensor(dist.reshape(N, N), dtype=torch.float32, device=xy.device)

    def _pairwise_bearing(self, xy):
        lat1, lon1 = torch.deg2rad(xy[:, 0])[:, None], torch.deg2rad(xy[:, 1])[:, None]
        lat2, lon2 = torch.deg2rad(xy[:, 0])[None, :], torch.deg2rad(xy[:, 1])[None, :]
        y = torch.sin(lon2 - lon1) * torch.cos(lat2)
        x = torch.cos(lat1) * torch.sin(lat2) - \
            torch.sin(lat1) * torch.cos(lat2) * torch.cos(lon2 - lon1)
        return torch.atan2(y, x)


class LocalPerformerBlock(nn.Module):
    def __init__(self, dim, heads=4, bias_heads=4, window_size=8, stride=None,
                 mlp_ratio=4, dropout=0.1, coordinates=None, device='cpu'):
        super().__init__()
        self.att = LocalPerformerAttention(dim, heads, bias_heads,
                                           window_size, stride,
                                           coordinates, device)
        hidden = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x, u10, v10):
        x = x + self.att(x, u10, v10)
        return x + self.ffn(x)


# --------------------------------------------------------------------------- #
# 2. Utility Layers (FCLayer / MLP)
# --------------------------------------------------------------------------- #
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        return self.conv(x) if self.conv else x


class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc1 = FCLayer(c_in, c_in // 2)
        self.fc2 = FCLayer(c_in // 2, c_out)
        nn.init.xavier_uniform_(self.fc1.linear.weight);
        nn.init.zeros_(self.fc1.linear.bias)
        nn.init.xavier_uniform_(self.fc2.linear.weight);
        nn.init.zeros_(self.fc2.linear.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x.permute(0, 3, 1, 2)))
        return self.fc2(x).permute(0, 2, 3, 1)


# --------------------------------------------------------------------------- #
# 3. Adaptive-K Sparse MoE-TCN (已去掉熵正则化)
# --------------------------------------------------------------------------- #
class AdaptiveSparseMoETemporalDecoderTCN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_experts: int = 4,
                 k_min: int = 1,
                 k_max: int = 4,
                 dropout: float = 0.1,
                 num_nodes: int = None,
                 embed_dim: int = None):
        super().__init__()
        assert num_nodes is not None, "`num_nodes` must be provided"
        self.E, self.k_min, self.k_max = num_experts, k_min, k_max
        embed_dim = embed_dim or hidden_dim

        kernels = [3, 3, 4, 4][:self.E]
        dilations = [1, 2, 4, 8][:self.E]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim,
                          kernel_size=k, dilation=d,
                          padding=(k - 1) * d),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k, d in zip(kernels, dilations)
        ])

        self.station_emb = nn.Embedding(num_nodes, embed_dim)

        gate_in_dim = hidden_dim + embed_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.E)
        )

        self.mlp_k = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.tau = 0.5

    def forward(self, enc, T_out, context, station_ids):
        Bn, L, C = enc.shape
        x = enc.transpose(1, 2)

        s_emb = self.station_emb(station_ids)
        gate_in = torch.cat([context, s_emb], dim=-1)

        logits = self.gate(gate_in) / self.tau

        k_raw = self.mlp_k(gate_in).squeeze(-1)
        k_norm = torch.sigmoid(k_raw)
        K_dyn = (k_norm * (self.k_max - self.k_min) + self.k_min) \
            .round().long().clamp(self.k_min, self.k_max)

        max_k = self.k_max
        top_vals, top_idx = logits.topk(max_k, dim=-1)
        mask = torch.arange(max_k, device=logits.device) \
                   .unsqueeze(0) < K_dyn.unsqueeze(1)
        top_vals = top_vals.masked_fill(~mask, float('-1e9'))
        w_top = F.softmax(top_vals, dim=-1) * mask.float()

        exp_stack = torch.stack([exp(x)[..., -T_out:] for exp in self.experts], dim=1)
        w_full = torch.zeros_like(exp_stack[:, :, :1, :])
        T = exp_stack.size(-1)
        idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, T)
        src = w_top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, T)
        w_full.scatter_(1, idx, src)

        mixed = (exp_stack * w_full).sum(dim=1)

        return mixed.transpose(1, 2)  # 仅返回 dec 特征，不再返回 ent_reg


# --------------------------------------------------------------------------- #
# 4. Temporal / Spatial Conv Layer
# --------------------------------------------------------------------------- #
class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="GLU"):
        super().__init__()
        self.kt, self.act, self.c_out = kt, act, c_out
        self.align = Align(c_in, c_out) if c_in != c_out else None
        self.conv = nn.Conv2d(c_in, c_out * 2 if act == "GLU" else c_out, (kt, 1))

    def forward(self, x):
        xin = self.align(x) if self.align else x
        xin = xin[:, :, self.kt - 1:, :]
        xcv = self.conv(x)
        if self.act == "GLU":
            v, g = xcv.split(self.c_out, 1)
            return (v + xin) * torch.sigmoid(g)
        return F.relu(xcv + xin)


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, alpha=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.Tensor(c_in, c_out, ks))
        self.bias = nn.Parameter(torch.Tensor(1, c_out, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, x, Lk):
        xc = torch.einsum('knm,bclm->bclkn', Lk, x)
        xgc = torch.einsum('iok,bclkn->boln', self.theta, xc) + self.bias
        return F.relu(torch.sigmoid(self.alpha) * x + (1 - torch.sigmoid(self.alpha)) * xgc)


# --------------------------------------------------------------------------- #
# 5. ASMSTCN
# --------------------------------------------------------------------------- #
class ASMSTCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes, input_length,
                 input_dim, hidden_dim, output_dim, output_length,
                 dropout=0.1, device='cpu', coordinates=None):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.out_len = output_length
        A = adj_matrix.clone().to(device)
        L = torch.eye(num_nodes, device=device) - A
        Lk = [torch.eye(num_nodes, device=device), L.clone()]
        for _ in range(2, 3):
            Lk.append(2 * L @ Lk[-1] - Lk[-2])
        self.register_buffer('Lk', torch.stack(Lk, 0))

        mid = hidden_dim // 2
        self.tconv11 = TemporalConvLayer(3, input_dim, mid)
        self.sconv12 = SpatioConvLayer(3, mid, mid)
        self.tconv13 = TemporalConvLayer(3, mid, mid)
        self.ln1 = nn.LayerNorm([num_nodes, mid])
        self.dp1 = nn.Dropout(dropout)
        self.local_att = LocalPerformerBlock(
            dim=mid, heads=4, bias_heads=4,
            window_size=8, stride=8,
            coordinates=coordinates, device=device)

        self.decoder_tcn = AdaptiveSparseMoETemporalDecoderTCN(
            hidden_dim=mid, num_experts=4,
            k_min=1, k_max=4, dropout=dropout,
            num_nodes=num_nodes, embed_dim=mid
        )
        self.mlp = MLP(mid * 3, output_dim)

    def forward(self, x):
        B, T, N, Fdim = x.shape
        assert Fdim == 13, f"Expected 13 features, got {Fdim}"
        u10 = torch.nan_to_num(x[..., 1], nan=0.0)
        v10 = torch.nan_to_num(x[..., 2], nan=0.0)

        h = self.tconv11(x.permute(0, 3, 1, 2))
        h = self.sconv12(h, self.Lk)
        h = self.tconv13(h).permute(0, 2, 3, 1)
        h = self.dp1(self.ln1(h))

        h = self.local_att(h, u10[:, :h.size(1)], v10[:, :h.size(1)])

        B2, L1, N2, C = h.shape
        enc = h.permute(0, 2, 1, 3).reshape(B2 * N2, L1, C)
        context = enc.mean(dim=1)
        station_ids = torch.arange(N2, device=x.device).repeat(B2).long()

        dec = self.decoder_tcn(enc, self.out_len, context, station_ids)  # 接收单个返回值
        dec = dec.view(B2, N2, self.out_len, C).permute(0, 2, 1, 3)
        g = h.mean(1).unsqueeze(1).expand(-1, self.out_len, -1, -1)
        pred = self.mlp(torch.cat([dec, g, g], dim=3)).squeeze(-1)

        return pred, h, dec

    # --------------------------------------------------------------------------- #


# 6. SpatialRegModule & ASMSTCN_Reg
# --------------------------------------------------------------------------- #
class SpatialRegModule(nn.Module):
    def __init__(self, c_in, n_proto, lap_weight,
                 adj_matrix, tau=1.0, tau_dyn=0.5,
                 alpha_init=0.1, device=None):
        super().__init__()
        self.tau_dyn = tau_dyn
        self.lap_w = lap_weight
        self.tau_pat = tau

        self.prototypes = nn.Parameter(torch.randn(n_proto, c_in))

        self.register_buffer("adj", adj_matrix.clone().float().to(device))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, h_node, dec_node):
        B, N, C = h_node.shape

        H_feat = F.normalize(h_node, dim=-1)
        D_feat = F.normalize(dec_node, dim=-1)
        dyn = F.softmax((H_feat @ D_feat.transpose(1, 2)) / self.tau_dyn, dim=-1)
        a = torch.clamp(self.alpha, 0.0, 1.0)
        Aeff = self.adj * (1 - a) + dyn * a

        dist = torch.cdist(h_node, self.prototypes.unsqueeze(0))
        pi = F.softmax(-dist.pow(2) / self.tau_pat, dim=-1)
        h_prime = h_node - torch.matmul(pi, self.prototypes)

        l_pat = h_prime.pow(2).mean()
        diff = h_prime - torch.matmul(Aeff, h_prime)
        l_lap = diff.pow(2).mean()

        return l_pat + self.lap_w * l_lap


class ASMSTCN_Reg(nn.Module):
    def __init__(self, adj_matrix, num_nodes,
                 input_length, input_dim, hidden_dim,
                 output_dim, output_length, dropout=0.1,
                 nmb_prototype=50, lap_reg_weight=0.1,
                 device='cpu', coordinates=None,
                 tau_dyn=0.5, alpha_init=0.1):
        super().__init__()
        self.base = ASMSTCN(adj_matrix, num_nodes,
                            input_length, input_dim,
                            hidden_dim, output_dim,
                            output_length, dropout,
                            device, coordinates)
        self.spatial_reg = SpatialRegModule(
            c_in=hidden_dim // 2, n_proto=nmb_prototype,
            lap_weight=lap_reg_weight, adj_matrix=adj_matrix,
            tau=1.0, tau_dyn=tau_dyn,
            alpha_init=alpha_init, device=device)

    def forward(self, x):
        pred, h, dec = self.base(x)
        if self.training:
            loss_space = self.spatial_reg(h.mean(1), dec.mean(1))
            return pred, loss_space
        return pred