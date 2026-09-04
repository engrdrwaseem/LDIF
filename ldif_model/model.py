"""
LDIF: Latent Dual Interaction Flow – PyTorch implementation.

Provides LDIFBlock, LDIFModel, LDIFStatic, and LDIFSequential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Performance tweaks
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
else:
    try:
        n_threads = max(1, os.cpu_count() - 1)
        torch.set_num_threads(n_threads)
    except:
        pass


class LDIFBlock(nn.Module):
    """
    Single LDIF layer with full gate.
    """
    def __init__(self, d, k_max, r, gamma=0.01, z_init_mean=1.0, z_init_std=0.01):
        super().__init__()
        self.d = d
        self.k_max = k_max
        self.r = r
        self.gamma = gamma

        # Spectrum: s = sigma(z)
        self.z = nn.Parameter(torch.randn(k_max) * z_init_std + z_init_mean)

        # Symmetric branch
        U_S_raw = torch.empty(k_max, d)
        V_S_raw = torch.empty(k_max, d)
        nn.init.kaiming_uniform_(U_S_raw, a=math.sqrt(5))
        nn.init.kaiming_uniform_(V_S_raw, a=math.sqrt(5))
        self.U_S = nn.Parameter(U_S_raw.T)
        self.V_S = nn.Parameter(V_S_raw.T)

        # Skew-symmetric branch
        U_A_raw = torch.empty(k_max, d)
        V_A_raw = torch.empty(k_max, d)
        nn.init.kaiming_uniform_(U_A_raw, a=math.sqrt(5))
        nn.init.kaiming_uniform_(V_A_raw, a=math.sqrt(5))
        self.U_A = nn.Parameter(U_A_raw.T)
        self.V_A = nn.Parameter(V_A_raw.T)

        # Gate projections
        V_h_raw = torch.empty(r, d)
        V_Sr_raw = torch.empty(r, d)
        V_Ar_raw = torch.empty(r, d)
        nn.init.kaiming_uniform_(V_h_raw, a=math.sqrt(5))
        nn.init.kaiming_uniform_(V_Sr_raw, a=math.sqrt(5))
        nn.init.kaiming_uniform_(V_Ar_raw, a=math.sqrt(5))
        self.V_h = nn.Parameter(V_h_raw.T)
        self.V_Sr = nn.Parameter(V_Sr_raw.T)
        self.V_Ar = nn.Parameter(V_Ar_raw.T)

        U_h_raw = torch.empty(d, r)
        U_Sr_raw = torch.empty(d, r)
        U_Ar_raw = torch.empty(d, r)
        std_h = 1.0 / math.sqrt(r)
        bound_h = math.sqrt(3.0) * std_h
        U_h_raw.uniform_(-bound_h, bound_h)
        U_Sr_raw.uniform_(-bound_h, bound_h)
        U_Ar_raw.uniform_(-bound_h, bound_h)
        self.U_h = nn.Parameter(U_h_raw)
        self.U_Sr = nn.Parameter(U_Sr_raw)
        self.U_Ar = nn.Parameter(U_Ar_raw)
        self.b_g = nn.Parameter(torch.zeros(d))

        # Bounded step-size parameters
        self.alpha_raw = nn.Parameter(torch.ones(d))
        self.lambda_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def s(self):
        return torch.sigmoid(self.z)

    @property
    def S2(self):
        return self.s ** 2

    @property
    def alpha(self):
        return torch.clamp(F.softplus(self.alpha_raw), min=0.5, max=2.0)

    @property
    def lambda_(self):
        return 0.05 + 0.45 * torch.sigmoid(self.lambda_raw)

    def _ensure_batch(self, h):
        if h.dim() == 1:
            return h.unsqueeze(0), True
        return h, False

    def _restore_batch(self, h, was_1d):
        if was_1d:
            return h.squeeze(0)
        return h

    def forward(self, h):
        h, was_1d = self._ensure_batch(h)
        s2 = self.S2

        # Symmetric response
        U_S_h = torch.matmul(h, self.U_S)
        V_S_h = torch.matmul(h, self.V_S)
        u_scaled = U_S_h * s2.unsqueeze(0)
        v_scaled = V_S_h * s2.unsqueeze(0)
        r_S = torch.matmul(u_scaled, self.V_S.T) + torch.matmul(v_scaled, self.U_S.T)

        # Skew-symmetric response
        U_A_h = torch.matmul(h, self.U_A)
        V_A_h = torch.matmul(h, self.V_A)
        uA_scaled = U_A_h * s2.unsqueeze(0)
        vA_scaled = V_A_h * s2.unsqueeze(0)
        r_A = torch.matmul(uA_scaled, self.V_A.T) - torch.matmul(vA_scaled, self.U_A.T)

        # Gating
        gate_in = torch.matmul(h, self.V_h) @ self.U_h.T
        gate_in += torch.matmul(r_S, self.V_Sr) @ self.U_Sr.T
        gate_in += torch.matmul(r_A, self.V_Ar) @ self.U_Ar.T
        gate_in += self.b_g
        g = torch.sigmoid(gate_in)
        R = g * r_S + (1 - g) * r_A

        # Master update
        lambda_ = self.lambda_
        alpha = self.alpha
        h_next = h + lambda_ * alpha * torch.tanh(R) - lambda_ * self.gamma * h

        intermediates = {
            'r_S': r_S, 'r_A': r_A, 'gate': g, 'R': R,
            's': self.s, 'alpha': alpha, 'lambda': lambda_,
            'z': self.z, 'h_next': h_next,
        }
        return self._restore_batch(h_next, was_1d), intermediates

    def spectrum_penalty(self):
        return torch.sum(self.s)

    def mean_spectrum(self):
        return self.s.mean().item()

    def effective_rank(self, s_threshold=0.1):
        return (self.s > s_threshold).sum().item()


class LDIFModel(nn.Module):
    """
    Base LDIF model supporting both static and sequential data.
    """
    def __init__(self, input_dim, output_dim, task_type,
                 num_layers=2, k_max=16, r=8, gamma=0.01,
                 input_scale=2.0, z_init_mean=1.0, z_init_std=0.01,
                 use_positional_encoding=True, use_random_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.use_random_init = use_random_init

        self.input_proj = nn.Linear(input_dim, input_dim)
        self.input_scale = nn.Parameter(torch.ones(1) * input_scale)

        if use_positional_encoding:
            self.pos_embed = nn.Linear(1, input_dim)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
            nn.init.zeros_(self.pos_embed.bias)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            LDIFBlock(input_dim, k_max, r,
                      gamma=gamma,
                      z_init_mean=z_init_mean,
                      z_init_std=z_init_std)
            for _ in range(num_layers)
        ])

        if task_type == 'regression':
            self.head = nn.Linear(input_dim, 1)
        elif task_type in ['binary_classification', 'imbalanced_binary']:
            self.head = nn.Linear(input_dim, 1)
        elif task_type == 'multiclass':
            self.head = nn.Linear(input_dim, output_dim)
        else:
            self.head = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x, return_intermediates=False):
        is_sequential = x.dim() == 3
        all_intermediates = {}

        if not is_sequential:
            h = x
            for idx, block in enumerate(self.blocks):
                h, inter = block(h)
                if return_intermediates:
                    all_intermediates[f'layer_{idx}'] = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in inter.items()}
            out = self.head(h)
            if self.task_type in ['regression', 'binary_classification', 'imbalanced_binary']:
                out = out.view(-1)
            return (out, all_intermediates) if return_intermediates else out

        # Sequential path
        B, T, D = x.shape
        if self.use_random_init:
            h = torch.randn(B, D, device=x.device) * 0.01
        else:
            h = torch.zeros(B, D, device=x.device)

        if self.use_positional_encoding and self.pos_embed is not None:
            time_indices = torch.arange(1, T+1, device=x.device).float().view(1, T, 1)
            pos_enc = self.pos_embed(time_indices)
            pos_enc_exp = pos_enc.expand(B, -1, -1)
        else:
            pos_enc_exp = None

        for t in range(T):
            x_t = x[:, t, :]
            if pos_enc_exp is not None:
                x_t = x_t + pos_enc_exp[:, t, :]
            x_t = torch.tanh(self.input_proj(x_t))
            h = h + self.input_scale * x_t

            for idx, block in enumerate(self.blocks):
                h, inter = block(h)
                if return_intermediates and (t == 0 or t == T-1):
                    time_tag = 't0' if t == 0 else 't_end'
                    all_intermediates[f'{time_tag}_layer_{idx}'] = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in inter.items()
                    }
        out = self.head(h)
        if self.task_type in ['regression', 'binary_classification', 'imbalanced_binary']:
            out = out.view(-1)
        return (out, all_intermediates) if return_intermediates else out

    def spectrum_penalty(self):
        return torch.stack([block.spectrum_penalty() for block in self.blocks]).sum()

    def get_mean_spectrum(self):
        return [block.mean_spectrum() for block in self.blocks]


class LDIFStatic(LDIFModel):
    """
    LDIF for static (tabular) data with recommended defaults.
    """
    def __init__(self, input_dim, output_dim, task_type,
                 num_layers=2, k_max=16, r=8, gamma=0.01):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            num_layers=num_layers,
            k_max=k_max,
            r=r,
            gamma=gamma,
            input_scale=2.0,
            z_init_mean=1.0,
            z_init_std=0.01,
            use_positional_encoding=False,
            use_random_init=True
        )


class LDIFSequential(LDIFModel):
    """
    LDIF for sequential (time-series) data with recommended defaults.
    """
    def __init__(self, input_dim, output_dim, task_type,
                 num_layers=2, k_max=16, r=8, gamma=0.2):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            num_layers=num_layers,
            k_max=k_max,
            r=r,
            gamma=gamma,
            input_scale=0.05,
            z_init_mean=-2.0,
            z_init_std=0.01,
            use_positional_encoding=True,
            use_random_init=False
        )


def compute_mu(N_train, mu_factor=1.0):
    """
    Compute L1 penalty coefficient using: mu = 1.82 * N_train^(-0.55) * mu_factor
    """
    return 1.82 * (N_train ** (-0.55)) * mu_factor


def compile_model(model, mode='default'):
    """
    Compile the model with torch.compile for faster execution.
    """
    if hasattr(torch, 'compile'):
        try:
            return torch.compile(model, mode=mode, fullgraph=False)
        except Exception as e:
            print(f"torch.compile failed: {e}. Using uncompiled model.")
            return model
    else:
        return model
