import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Callable
from functools import partial
import math
import torch.utils.checkpoint as checkpoint

import numbers
import warnings

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

try:
    from .utils.csm_triton import cross_scan_fn, cross_merge_fn
except:
    from utils.csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .utils.csms6s import selective_scan_fn_vmamba, selective_scan_flop_jit
except:
    from utils.csms6s import selective_scan_fn_vmamba, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn



class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape

        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C

        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

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

        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):

        h, w = x.shape[-2:]

        return to_4d(self.body(to_3d(x)), h, w)


class PModule(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1)
        #self.selayer = SELayer(hidden_dim//2)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim//2, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        hh,ww = img_size[0],img_size[1]
        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)

        x1,x2 = self.dwconv(x).chunk(2, dim=1)
        x3 = x1 * x2
        #x4=self.selayer(x3)
        # flaten
        x3 = rearrange(x3, ' b c h w -> b (h w) c', h=hh, w=ww)
        y = self.linear2(x3)
        y = rearrange(y, ' b (h w) c -> b h w c', h=hh, w=ww)
        return y

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim//2
        # fc2
        flops += H * W * self.hidden_dim//2 * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        return flops


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

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
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias



class SS2D_Global(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=32,
            d_state=16,
            expand=2,
            dt_rank="auto",
            act_layer = nn.SiLU,
            # dwconv ===============
            d_conv=3,
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            channel_first = False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()


        self.k_group = 4
        self.d_model = d_model
        self.d_state = d_state

        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first

        # in proj =======================================
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()


        # conv =======================================
        self.d_conv = d_conv
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj




        # out proj =======================================
        self.oact = False
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            k_group=self.k_group,
        )



    def forward_core(
        self,
        x: torch.Tensor=None,
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,):

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)


        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W
        ssoflex = True

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn_vmamba(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend="oflex")


        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()  # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y= self.forward_core(x)

        y = self.out_act(y)
        y = y * F.silu(z)


        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class SS2D_region(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

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

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


        dc_inner = 4
        self.KC = 2
        self.dtc_rank = 6 #6
        self.dc_state = 16 #16
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) # (K, N, inner)
        del self.xc_proj
        self.Dsc = nn.Parameter(torch.ones((self.KC * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.KC * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner)))
        self.channel_norm = LayerNorm(self.d_inner, LayerNorm_type='WithBias')

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

    @staticmethod
    def sort_pixels_by_mask(original_image, mask, class_order=[0, 1]):
        """
        Sort pixels in the image based on the mask.
        Background pixels first, then foreground pixels.

        Args:
            original_image (torch.Tensor): [B, C, H, W]
            mask (torch.Tensor): [B, 1, H, W], binary mask where 1 = foreground
            class_order (list): Order of classes to sort, e.g., [0,1] for background first

        Returns:
            sorted_pixels (torch.Tensor): [B, C, H*W]
            sorted_indices (torch.Tensor): [B, C, H*W]
        """
        B, C, H, W = original_image.shape

        # 1. 扁平化图像和掩码
        original_flat = original_image.view(B, C, H * W)  # [B, C, H*W]
        mask_flat = mask.view(B, 1, H * W)  # [B, 1, H*W]

        # 2. 广播掩码到通道维度
        mask_broadcast = mask_flat.expand(-1, C, -1)  # [B, C, H*W]

        # 3. 生成排序键
        # 根据 class_order 定义排序优先级
        num_classes = max(class_order) + 1
        class_order_map = torch.full((num_classes,), len(class_order), dtype=torch.long, device=original_image.device)
        for idx, cls in enumerate(class_order):
            class_order_map[cls] = idx

        # 应用 class_order_map 到 mask，生成 primary_sort_key
        # mask 的值为0或1，对应 class_order_map[0]和class_order_map[1]
        primary_sort_key = class_order_map[mask_flat.long()]  # [B, 1, H*W]

        # 不要squeeze，而是保持 [B, 1, H*W]，以便后续扩展
        primary_sort_key = primary_sort_key.expand(-1, C, -1)  # [B, C, H*W]

        # 生成像素索引作为 secondary_sort_key
        pixel_indices = torch.arange(H * W, device=original_image.device).view(1, 1, H * W).expand(B, C,
                                                                                                   H * W)  # [B, C, H*W]

        # 定义 multiplier，确保 primary_sort_key 优先于 secondary_sort_key
        multiplier = H * W  # 例如, H*W=36

        # 组合排序键
        combined_sort_key = primary_sort_key * multiplier + pixel_indices  # [B, C, H*W]

        # 4. 排序
        sorted_indices = combined_sort_key.argsort(dim=2)  # [B, C, H*W]

        # 5. 重排像素
        # torch.gather 的 index 需要与 original_flat 在 dim=2 上匹配
        sorted_pixels = torch.gather(original_flat, dim=2, index=sorted_indices)  # [B, C, H*W]

        return sorted_pixels
    @staticmethod
    def local_scan(x, H=14, W=14, w=10, flip=False, column_first=False):
        """Local windowed scan in LocalMamba
          Input:
              x: [B, C, H, W]
              H, W: original width and height
              column_first: column-wise scan first (the additional direction in VMamba)
          Return: [B, C, L]
          """
        B, C, _, _ = x.shape

        x = x.view(B, C, H, W)
        Hg, Wg = math.floor(H / w), math.floor(W / w)
        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = x[:, :, :newH, :newW]
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x

    def forward_core(self, x: torch.Tensor, xm: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 窗口大小，块数

        recon_img = self.sort_pixels_by_mask(x, xm, class_order=[0, 1])

        x_hwwh = torch.stack([recon_img.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)

        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def channelforward_core(self, xc: torch.Tensor):
        xc = xc.permute(0,3,1,2).contiguous()
        xc = self.pooling(xc) #b,d,1,1

        xc = xc.permute(0,2,1,3).contiguous() #b,1,d,1

        xc = self.conv_cin(xc) #b,4,d,1

        xc = xc.squeeze(-1) #b,4,d

        B, D, L = xc.shape #b,1,c
        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1)

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)  # 8,2,1,96; 2,38,1 ->8,2,38,96

        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2)  # 8,2,38,96-> 6,16,16
        # dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()
        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2
        out_y = self.selective_scan(
            xsc, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        y = y.unsqueeze(-1) # b,4,d,1

        y = self.conv_cout(y) # b,1,d,1

        y = y.transpose(dim0=1, dim1=2).contiguous() # b,d,1,1

        y = self.channel_norm(y)
        y = y.to(xc.dtype)
        return y


    def forward(self, x: torch.Tensor, xm:torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x, xm)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        y = y * F.silu(z)

        # c = self.channelforward_core(y).permute(0,2,3,1).contiguous()
        #
        # yy = y * c
        # yy = y + yy
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D_localrs(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

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

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


        dc_inner = 4
        self.KC = 2
        self.dtc_rank = 6 #6
        self.dc_state = 16 #16
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) # (K, N, inner)
        del self.xc_proj
        self.Dsc = nn.Parameter(torch.ones((self.KC * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.KC * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner)))
        self.channel_norm = LayerNorm(self.d_inner, LayerNorm_type='WithBias')

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

    @staticmethod
    def classify_windows(windows, w):

        sums = windows.sum(dim=-1)  # [B, C, Hg*Wg]
        all_zeros = (sums == 0)
        all_ones = (sums == w * w)

        # 默认分类=2(混合)
        classifications = torch.full_like(sums, 2, dtype=torch.long)
        classifications[all_zeros] = 0
        classifications[all_ones] = 1

        return classifications

    @staticmethod
    def reorder_image_by_windows(image_windows, classifications, class_order=None):
        B, C, HgWg, ww = image_windows.shape

        if class_order is None:
            class_order = [0, 1, 2]

        class_order_map = torch.zeros(3, dtype=torch.long, device=classifications.device)

        for new_order, cls in enumerate(class_order):
            class_order_map[cls] = new_order

        sorting_key = class_order_map[classifications]  # [B, C, Hg*Wg]


        sorted_indices = sorting_key.argsort(dim=-1, stable=True)  # [B, C, Hg*Wg]

        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(B, C, -1,
                                                                      image_windows.size(-1))  # [B, C, Hg*Wg, w*w]

        # 使用 gather 进行重排
        sorted_windows = torch.gather(image_windows, dim=2, index=sorted_indices_expanded)  # [B, C, Hg*Wg, w*w]

        return sorted_windows

    @staticmethod
    def local_scan(x, H=14, W=14, w=10, flip=False, column_first=False):
        """Local windowed scan in LocalMamba
          Input:
              x: [B, C, H, W]
              H, W: original width and height
              column_first: column-wise scan first (the additional direction in VMamba)
          Return: [B, C, L]
          """
        B, C, _, _ = x.shape

        x = x.view(B, C, H, W)
        Hg, Wg = math.floor(H / w), math.floor(W / w)
        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = x[:, :, :newH, :newW]
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x

    def forward_core(self, x: torch.Tensor, xm: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 窗口大小，块数
        w = 10
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)

        #对mask分块
        mask_l = xm.view(B, 1, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, 1, Hg * Wg, w * w)

        #对每个块分类   全0： {0}  Non-shadow   全1： {1} Shadow     有1有0： {2}   边界区域
        window_classifications = self.classify_windows(mask_l, w)



        #将img按照分块规则划分窗口
        image_windows = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5)

        image_windows = image_windows.reshape(B, C, Hg * Wg, w * w)

        #根据得到的位置对img的窗口重排
        recon_img = self.reorder_image_by_windows(image_windows, window_classifications, class_order=[0, 2, 1])

        x1 = self.local_scan(recon_img, H, W, w)
        x2 = self.local_scan(recon_img, H, W, w, column_first=True)
        x3 = self.local_scan(recon_img, H, W, w,  flip=True)
        x4 = self.local_scan(recon_img, H, W, w, column_first=True, flip=True)

        xs = torch.stack([x1, x2, x3, x4], dim=1)

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)

        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def channelforward_core(self, xc: torch.Tensor):
        xc = xc.permute(0,3,1,2).contiguous()
        xc = self.pooling(xc) #b,d,1,1

        xc = xc.permute(0,2,1,3).contiguous() #b,1,d,1

        xc = self.conv_cin(xc) #b,4,d,1

        xc = xc.squeeze(-1) #b,4,d

        B, D, L = xc.shape #b,1,c
        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1)

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)  # 8,2,1,96; 2,38,1 ->8,2,38,96

        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2)  # 8,2,38,96-> 6,16,16
        # dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()
        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2
        out_y = self.selective_scan(
            xsc, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        y = y.unsqueeze(-1) # b,4,d,1

        y = self.conv_cout(y) # b,1,d,1

        y = y.transpose(dim0=1, dim1=2).contiguous() # b,d,1,1

        y = self.channel_norm(y)
        y = y.to(xc.dtype)
        return y


    def forward(self, x: torch.Tensor, xm:torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x, xm)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out





class LocalRSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.,
            attn_drop_rate: float = 0.,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        self.ln_1 = LayerNorm(hidden_dim, 'WithBias')
        self.attention_localrs = SS2D_localrs(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = LayerNorm(hidden_dim, 'WithBias')
        self.skip_scale2= nn.Parameter(torch.ones(hidden_dim))
        # self.ffn = FeedForward(hidden_dim, ffn_expansion_factor=2.5, bias=False)
        mlp_hidden_dim = int(hidden_dim * 4)
        self.Pmodule = PModule(dim=hidden_dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)



    def forward(self, input, xm, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        xx = input.permute(0, 3, 1, 2).contiguous()  # [B,H,W,C]
        x1 = self.ln_1(xx)
        H = x_size[0]
        W = x_size[1]
        x1 = x1.permute(0, 2, 3, 1).contiguous()


        RB = self.attention_localrs(x1, xm)


        x = input + self.drop_path(RB)

        c = self.ln_2(x.permute(0, 3, 1, 2).contiguous())
        d = c.permute(0, 2, 3, 1).contiguous()
        e = d.view(B, -1, C).contiguous()

        x = x + self.drop_path(self.Pmodule(e, img_size=x_size))
        # x = x + self.ffn(self.ln_2(x.permute(0, 3, 1, 2).contiguous())).permute(0, 2, 3, 1).contiguous()

        x = x.view(B, -1, C).contiguous()

        return x



class RegionBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.,
            attn_drop_rate: float = 0.,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        self.ln_1 = LayerNorm(hidden_dim, 'WithBias')
        self.attention_region = SS2D_region(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = LayerNorm(hidden_dim, 'WithBias')
        self.skip_scale2= nn.Parameter(torch.ones(hidden_dim))
        # self.ffn = FeedForward(hidden_dim, ffn_expansion_factor=2.5, bias=False)
        mlp_hidden_dim = int(hidden_dim * 4)
        self.Pmodule = PModule(dim=hidden_dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)



    def forward(self, input, xm, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        xx = input.permute(0, 3, 1, 2).contiguous()  # [B,H,W,C]
        x1 = self.ln_1(xx)
        H = x_size[0]
        W = x_size[1]

        x1 = x1.permute(0, 2, 3, 1).contiguous()


        Reg = self.attention_region(x1, xm)


        x = input + self.drop_path(Reg)

        c = self.ln_2(x.permute(0, 3, 1, 2).contiguous())
        d = c.permute(0, 2, 3, 1).contiguous()
        e = d.view(B, -1, C).contiguous()

        x = x + self.drop_path(self.Pmodule(e, img_size=x_size))
        # x = x + self.ffn(self.ln_2(x.permute(0, 3, 1, 2).contiguous())).permute(0, 2, 3, 1).contiguous()

        x = x.view(B, -1, C).contiguous()

        return x


class GlobalBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.,
            attn_drop_rate: float = 0.,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        self.ln_1 = LayerNorm(hidden_dim, 'WithBias')
        self.attention_global = SS2D_Global(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = LayerNorm(hidden_dim, 'WithBias')
        self.skip_scale2= nn.Parameter(torch.ones(hidden_dim))
        # self.ffn = FeedForward(hidden_dim, ffn_expansion_factor=2.5, bias=False)
        mlp_hidden_dim = int(hidden_dim * 4)
        self.Pmodule = PModule(dim=hidden_dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)



    def forward(self, input, xm, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        xx = input.permute(0, 3, 1, 2).contiguous()  # [B,H,W,C]
        x1 = self.ln_1(xx)
        H = x_size[0]
        W = x_size[1]

        x1 = x1.permute(0, 2, 3, 1).contiguous()


        glo = self.attention_global(x1)


        x = input + self.drop_path(glo)

        c = self.ln_2(x.permute(0, 3, 1, 2).contiguous())
        d = c.permute(0, 2, 3, 1).contiguous()
        e = d.view(B, -1, C).contiguous()

        x = x + self.drop_path(self.Pmodule(e, img_size=x_size))
        # x = x + self.ffn(self.ln_2(x.permute(0, 3, 1, 2).contiguous())).permute(0, 2, 3, 1).contiguous()

        x = x.view(B, -1, C).contiguous()

        return x

#########################################
########### Basic layer of ShadowFormer ################
class BasicShadowMamba(nn.Module):
    def __init__(self, dim, depth, d_state = 16,
                 mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,cab=int):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        if cab == 1:
            self.blocks = nn.ModuleList([
                LocalRSBlock(hidden_dim=self.dim,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             attn_drop=attn_drop,
                             d_state=d_state,
                             expand=2,
                             )
                for i in range(depth)])
        if cab == 2:
            self.blocks = nn.ModuleList([
                RegionBlock(hidden_dim=self.dim,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             attn_drop=attn_drop,
                             d_state=d_state,
                             expand=2,
                             )
                for i in range(depth)])
        if cab == 3:
            self.blocks = nn.ModuleList([
                GlobalBlock(hidden_dim=self.dim,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             attn_drop=attn_drop,
                             d_state=d_state,
                             expand=2,
                             )
                for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def forward(self, x, xm, img_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, xm,img_size)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops





class ShadowMamba(nn.Module):
    def __init__(self,
                 in_chans=3,
                 out_channels=3,
                 dim=32,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 win_size=10,
                 mlp_ratio=2.,
                 d_state = 4,
                 drop_path_rate=0.1,
                 drop_rate=0.,
                 attn_drop=0.,
                 img_range=1.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint = False,
                 token_projection='linear',
                 token_mlp='PMFFN',
                 dowsample=Downsample, upsample=Upsample,
                 **kwargs):
        super().__init__()

        self.embed_dim = dim
        self.input_proj = InputProj(in_channel=4, out_channel=dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * dim+2, out_channel=in_chans, kernel_size=3, stride=1)

        self.mlp_ratio = mlp_ratio
        self.img_range = img_range
        self.patch_embed = InputProj(in_channel=4, out_channel=dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size

        base_d_state = d_state

        self.num_enc_layers = len(depths)//2
        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # Encoder
        self.encoderlayer_0 = BasicShadowMamba(dim=dim,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=1)
        self.dowsample_0 = dowsample(dim, dim * 2)
        self.encoderlayer_1 = BasicShadowMamba(dim=dim * 2,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=1)
        self.dowsample_1 = dowsample(dim * 2, dim * 4)
        self.encoderlayer_2 = BasicShadowMamba(dim=dim * 4,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=3)
        self.dowsample_2 = dowsample(dim * 4, dim * 8)
        # self.encoderlayer_3 = BasicShadowMamba(dim=dim * 8,
        #                     depth=depths[0],
        #                     d_state=base_d_state,
        #                     mlp_ratio=self.mlp_ratio,
        #                     drop=drop_rate,
        #                     attn_drop=attn_drop,
        #                     drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
        #                     norm_layer=norm_layer,
        #                     use_checkpoint=use_checkpoint,
        #                     cab=3)
        # self.dowsample_3 = dowsample(dim * 8, dim * 16)
        self.conv = BasicShadowMamba(dim=dim * 8,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=3)
        # self.upsample_0 = upsample(dim * 16, dim * 8)
        # self.decoderlayer_0 = BasicShadowMamba(dim=dim * 16,
        #                     depth=depths[0],
        #                     d_state=base_d_state,
        #                     mlp_ratio=self.mlp_ratio,
        #                     drop=drop_rate,
        #                     attn_drop=attn_drop,
        #                     drop_path=dec_dpr[:depths[5]],
        #                     norm_layer=norm_layer,
        #                     use_checkpoint=use_checkpoint,
        #                     cab=3)
        self.upsample_1 = upsample(dim * 8, dim * 4)
        self.decoderlayer_1 = BasicShadowMamba(dim=dim * 8,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=3)
        self.upsample_2 = upsample(dim * 8, dim * 2)
        self.decoderlayer_2 = BasicShadowMamba(dim=dim * 4,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=1)
        self.upsample_3 = upsample(dim * 4, dim * 1)
        self.decoderlayer_3 = BasicShadowMamba(dim=dim * 2+1,
                            depth=depths[0],
                            d_state=base_d_state,
                            mlp_ratio=self.mlp_ratio,
                            drop=drop_rate,
                            attn_drop=attn_drop,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            cab=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, xm):
        # Input Projection

        H, W = (x.shape[2], x.shape[3])
        y = self.input_proj(torch.cat([x, xm], 1))
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y, xm, img_size=(H, W))
        pool0 = self.dowsample_0(conv0, img_size=(H, W))
        m = nn.MaxPool2d(2)
        xm1 = m(xm)
        conv1 = self.encoderlayer_1(pool0,xm1,img_size=(H // 2, W // 2))

        pool1 = self.dowsample_1(conv1, img_size=(H // 2, W // 2))
        xm2 = m(xm1)
        conv2 = self.encoderlayer_2(pool1,xm2, img_size=(H // 2 ** 2, W // 2 ** 2))
        pool2 = self.dowsample_2(conv2, img_size=(H // 2 ** 2, W // 2 ** 2))

        xm3 = m(xm2)
        # conv3 = self.encoderlayer_3(pool2, xm3, img_size=(H // 2 ** 3, W // 2 ** 3))
        # pool3 = self.dowsample_3(conv3, img_size=(H // 2 ** 3, W // 2 ** 3))

        # xm4 = m(xm3)
        # Bottleneck
        conv4 = self.conv(pool2, xm3, img_size=(H // 2 ** 3, W // 2 ** 3))

        # Decoder
        # up0 = self.upsample_0(conv4, img_size=(H // 2 ** 4, W // 2 ** 4))

        # deconv0 = torch.cat([up0, conv3], -1)

        # deconv0 = self.decoderlayer_0(deconv0, xm3, img_size=(H // 2 ** 3, W // 2 ** 3))

        up1 = self.upsample_1(conv4, img_size=(H // 2 ** 3, W // 2 ** 3))

        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, xm2, img_size=(H // 2 ** 2, W // 2 ** 2))

        up2 = self.upsample_2(deconv1, img_size=(H // 2 ** 2, W // 2 ** 2))
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2,xm1, img_size=(H // 2, W // 2))

        up3 = self.upsample_3(deconv2, img_size=(H // 2, W // 2))
        deconv3 = torch.cat([up3, conv0, xm.flatten(2).transpose(1, 2).contiguous()], -1)
        deconv3 = self.decoderlayer_3(deconv3,xm, img_size=(H, W))

        # Output Projection
        y = self.output_proj(torch.cat([deconv3, xm.flatten(2).transpose(1, 2).contiguous()], -1), img_size=(H, W))
        return x + y