# The implementation of OVSSM, which includes CMLFSM internally

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numbers
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class OVSSM(nn.Module):
    def __init__(
            self,
            d_model,
            window_size,
            d_state=16,
            d_conv=3,
            expand=2.66,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_vis = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_inf = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d_vis = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_vis = nn.SiLU()

        self.conv2d_inf = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_inf = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm_vis = nn.LayerNorm(self.d_inner)
        self.out_norm_inf = nn.LayerNorm(self.d_inner)
        self.out_proj_vis = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_inf = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def create_patches(self, image_tensor, w, order='ltr_utd'):

        B, C, H, W = image_tensor.shape
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)

        assert H % w == 0 and W % w == 0, f"图像尺寸({H}x{W})必须能被patch大小({w})整除"

        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous()
        patches = image_tensor.view(B, Hg, w, Wg, w, C)

        if order == 'ltr_utd':
            patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()

        elif order == 'rtl_dtu':
            patches = patches.flip(1, 3).permute(0, 5, 1, 3, 2, 4).flip(4, 5).contiguous()

        elif order == 'utd_ltr':
            patches = patches.permute(0, 5, 3, 1, 4, 2).contiguous()

        elif order == 'dtu_rtl':
            patches = patches.flip(1, 3).permute(0, 5, 3, 1, 4, 2).flip(4, 5).contiguous()

        else:
            raise ValueError(f"Unsupported order: {order}")

        return patches

    def get_scan(self, x_vis, x_inf, orders):
        output = []
        for order in orders:
            x_vis_patches = self.create_patches(x_vis, self.window_size, order)
            x_inf_patches2 = self.create_patches(x_inf, self.window_size, order)

            B, C, Hg, Wg, w1, w2 = x_vis_patches.shape
            x_vis_patches_flat = x_vis_patches.reshape(B, C, Hg, Wg, w1 * w2)
            x_inf_patches2_flat = x_inf_patches2.reshape(B, C, Hg, Wg, w1 * w2)

            merged_patches = torch.cat([x_vis_patches_flat.unsqueeze(-2), x_inf_patches2_flat.unsqueeze(-2)], dim=-2)
            merged_patches = merged_patches.reshape(B, C, Hg, Wg, -1)

            merged_sequence = merged_patches.reshape(B, C, -1).unsqueeze(1)
            output.append(merged_sequence)
        output = torch.cat(output, dim=1)
        return output

    def reconstruct_images(self, final_result, w, x_vis, orders):
        B, C, H, W = x_vis.shape
        L = H * W
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        patch_size = w * w
        sequences = [final_result[:, i] for i in range(4)]

        y_vis = None
        y_inf = None

        for i, order in enumerate(orders):
            seq = sequences[i].reshape(B, C, Hg, Wg, 2, patch_size)
            patches1 = seq[:, :, :, :, 0, :].reshape(B, C, Hg, Wg, w, w)
            patches2 = seq[:, :, :, :, 1, :].reshape(B, C, Hg, Wg, w, w)

            if order == 'ltr_utd':
                patches1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous()
                patches2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous()
                img1 = patches1.view(B, C, Hg * w, Wg * w)
                img2 = patches2.view(B, C, Hg * w, Wg * w)
            elif order == 'rtl_dtu':
                patches1 = patches1.flip(2, 3)
                patches2 = patches2.flip(2, 3)

                patches1 = patches1.flip(4, 5)
                patches2 = patches2.flip(4, 5)

                img1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg * w, Wg * w)
                img2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg * w, Wg * w)

            elif order == 'utd_ltr':
                patches1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous()
                patches2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous()
                img1 = patches1.view(B, C, Hg * w, Wg * w).transpose(-1, -2)
                img2 = patches2.view(B, C, Hg * w, Wg * w).transpose(-1, -2)

            elif order == 'dtu_rtl':
                patches1 = patches1.flip(2, 3)
                patches2 = patches2.flip(2, 3)

                patches1 = patches1.flip(4, 5)
                patches2 = patches2.flip(4, 5)

                img1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg * w, Wg * w).transpose(2, 3)
                img2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg * w, Wg * w).transpose(2, 3)

            if y_vis == None:
                y_vis = img1.contiguous().view(B, -1, L).contiguous()
                y_inf = img2.contiguous().view(B, -1, L).contiguous()
            else:
                y_vis = y_vis + img1.contiguous().view(B, -1, L).contiguous()
                y_inf = y_inf + img2.contiguous().view(B, -1, L).contiguous()

        return y_vis, y_inf

    def forward_core(self, x_vis, x_inf):
        B, C, H, W = x_vis.shape
        L = H * W * 2
        K = 4

        orders = ['ltr_utd', 'rtl_dtu', 'utd_ltr', 'dtu_rtl']
        xs = self.get_scan(x_vis, x_inf, orders)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return self.reconstruct_images(out_y, self.window_size, x_vis, orders)

    def forward(self, visible, infrared):
        x_vis = rearrange(visible, 'b c h w -> b h w c')
        x_inf = rearrange(infrared, 'b c h w -> b h w c')

        B, H, W, C = x_vis.shape

        xz_vis = self.in_proj_vis(x_vis)
        x_vis, z_vis = xz_vis.chunk(2, dim=-1)

        xz_inf = self.in_proj_vis(x_inf)
        x_inf, z_inf = xz_inf.chunk(2, dim=-1)

        x_vis = x_vis.permute(0, 3, 1, 2).contiguous()
        x_vis = self.act_vis(self.conv2d_vis(x_vis))

        x_inf = x_inf.permute(0, 3, 1, 2).contiguous()
        x_inf = self.act_inf(self.conv2d_inf(x_inf))

        y_vis, y_inf = self.forward_core(x_vis, x_inf)

        y_vis = torch.transpose(y_vis, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_inf = torch.transpose(y_inf, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y_vis = self.out_norm_vis(y_vis)
        y_inf = self.out_norm_inf(y_inf)

        y_vis = y_vis * F.silu(z_vis)
        y_inf = y_inf * F.silu(z_inf)

        out_vis = self.out_proj_vis(y_vis)
        out_inf = self.out_proj_inf(y_inf)
        out_vis = rearrange(out_vis, 'b h w c -> b c h w')
        out_inf = rearrange(out_inf, 'b h w c -> b c h w')
        return out_vis, out_inf


if __name__ == '__main__':
    x = torch.randn(1, 32, 128, 128).to('cuda')
    y = torch.randn(1, 32, 128, 128).to('cuda')
    model = OVSSM(d_model=32, window_size=16).to('cuda')
    with torch.no_grad():
        x, y = model(x, y)
    print(x.shape)
    print(y.shape)
