import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_ as call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    return call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class LinearMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden=None,
        dim_out=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()

        dim_hidden = dim_hidden or dim_in
        dim_out = dim_out or dim_in
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden=None,
        dim_out=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()

        dim_hidden = dim_hidden or dim_in
        dim_out = dim_out or dim_in
        self.fc1 = nn.Conv2d(dim_in, dim_hidden, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(dim_hidden, dim_out, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

        self.norm1 = nn.BatchNorm2d(dim_hidden)
        self.norm2 = nn.BatchNorm2d(dim_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class STConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=9,
        # expand_ratio=2,
    ) -> None:
        super().__init__()

        self.expand_conv = nn.Conv2d(dim, dim * 2, 1)
        self.spat_conv = self._make_conv_block(
            dim, dim, (1, kernel_size), 1, (0, kernel_size // 2)
        )
        self.temp_conv = self._make_conv_block(
            dim, dim, (kernel_size, 1), 1, (kernel_size // 2, 0)
        )
        self.shrink_conv = nn.Conv2d(dim * 2, dim, 1)

    def _make_conv_block(self, dim_in, dim_out, k, s, p):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, k, s, p),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.expand_conv(x)
        x_s, x_t = x.chunk(2, dim=1)
        x_s = self.spat_conv(x_s)
        x_t = self.temp_conv(x_t)
        x = torch.cat([x_s, x_t], dim=1)
        x = self.shrink_conv(x)
        return x


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        expand_ratio=2,
    ) -> None:
        super().__init__()

        dim_hidden = int(dim * expand_ratio)
        self.conv1 = self._make_conv_block(dim, dim_hidden, 1, 1, 0)
        self.dwconv = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size, 1, kernel_size // 2, groups=dim_hidden
        )
        self.conv2 = self._make_conv_block(dim_hidden, dim, 1, 1, 0)

    def _make_conv_block(self, dim_in, dim_out, k, s, p):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, k, s, p),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qkv_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_head_dim=None,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qkv_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shape = (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class STCAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_head_dim: int = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.mid_dim = head_dim // 2
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward_spatial(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, J, _ = q.shape
        q = rearrange(q, "b t j (h c) -> (b h t) j c", h=self.num_heads)
        k = rearrange(k, "b t j (h c) -> (b h t) c j", h=self.num_heads)
        v = rearrange(v, "b t j (h c) -> (b h t) j c", h=self.num_heads)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(x, "(b h t) j c -> b h t j c", b=B, t=T)
        return x

    def forward_temporal(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, J, _ = q.shape
        q = rearrange(q, "b t j (h c) -> (b h j) t c", h=self.num_heads)
        k = rearrange(k, "b t j (h c) -> (b h j) c t", h=self.num_heads)
        v = rearrange(v, "b t j (h c) -> (b h j) t c", h=self.num_heads)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(x, "(b h j) t c -> b h t j c", b=B, j=J)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c t j -> b t j c")
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, C, 3).permute(4, 0, 1, 2, 3)
        qkv_s, qkv_t = qkv.chunk(2, dim=-1)

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]

        x_s = self.forward_spatial(q_s, k_s, v_s)
        x_t = self.forward_temporal(q_t, k_t, v_t)

        x = torch.cat([x_s, x_t], dim=-1)
        x = rearrange(x, "b h t j c -> b t j (c h)")
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b t j c -> b c t j")
        return x


class STAPAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_compress: int = None,
        block_size=3,
        block_stride=None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qkv_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        attn_head_dim: int = None,
    ) -> None:
        super().__init__()

        block_stride = block_stride or block_size
        block_pad = (val // 2 for val in block_size)
        dim_compress = dim_compress or dim

        self.num_heads = num_heads
        head_dim = dim_compress // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qkv_scale or head_dim**-0.5

        self.qkv = nn.Sequential(
            nn.Conv2d(
                dim,
                all_head_dim * 3,
                block_size,
                block_stride,
                block_pad,
                bias=qkv_bias,
            ),
            # nn.Conv2d(dim, all_head_dim * 3, block_size, block_stride, bias=qkv_bias),
            nn.BatchNorm2d(all_head_dim * 3),
            nn.GELU(),
        )
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Conv2d(all_head_dim, dim, 1)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, J = x.shape
        qkv = self.qkv(x)
        H, W = qkv.shape[-2:]  # H: Compressed T, W: Compressed J
        qkv = qkv.reshape(B, 3, -1, H, W).permute(1, 0, 2, 3, 4)
        # qkv = qkv.reshape(B, -1, 3, H, W).permute(2, 0, 1, 3, 4)
        qkv = rearrange(
            qkv, "qkv b (h c) ct cj -> qkv b h (ct cj) c", qkv=3, h=self.num_heads
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # shape = (B, num_heads, num_tokens, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(x, "b h (ct cj) c -> b (h c) ct cj", ct=H, cj=W)
        x = F.interpolate(x, size=(T, J), mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STCBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_out_ratio: float = 1.0,
        qkv_bias: bool = False,
        qkv_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_head_dim: int = None,
    ) -> None:
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.mixer = STCAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qkv_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            attn_head_dim=attn_head_dim,
        )

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = ConvMLP(
            dim_in=dim,
            dim_hidden=mlp_hidden_dim,
            dim_out=mlp_out_dim,
            drop_rate=drop_rate,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class STAPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_compress: int = None,
        block_size: int = 3,
        block_stride: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_out_ratio: float = 1.0,
        qkv_bias: bool = False,
        qkv_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_head_dim: int = None,
    ) -> None:
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.mixer = STAPAttention(
            dim,
            dim_compress=dim_compress,
            block_size=block_size,
            block_stride=block_stride,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            attn_head_dim=attn_head_dim,
        )

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = ConvMLP(
            dim_in=dim,
            dim_hidden=mlp_hidden_dim,
            dim_out=mlp_out_dim,
            drop_rate=drop_rate,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class STMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_compress: int = None,
        block_size: int = 3,
        block_stride: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_out_ratio: float = 1.0,
        qkv_bias: bool = False,
        qkv_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_head_dim: int = None,
    ) -> None:
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.inner_mixer = STCAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qkv_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            attn_head_dim=attn_head_dim,
        )
        self.outer_mixer = STAPAttention(
            dim,
            dim_compress=dim_compress,
            block_size=block_size,
            block_stride=block_stride,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            attn_head_dim=attn_head_dim,
        )

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = ConvMLP(
            dim_in=dim,
            dim_hidden=mlp_hidden_dim,
            dim_out=mlp_out_dim,
            drop_rate=drop_rate,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward_mixer(self, x):
        x_inner = self.inner_mixer(x)
        x_outer = self.outer_mixer(x)
        out = x_inner + x_outer
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.forward_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        dim_in,
        dim_feat,
        dim_compress=None,
        block_size=3,
        block_stride=None,
        mlp_ratio=4.0,
        mlp_out_ratio=1.0,
        num_heads=8,
        qkv_bias=False,
        qkv_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_head_dim=None,
        init_std=0.02,
        num_frames=243,
        num_joints=17,
    ):
        super().__init__()

        self.input_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, num_frames, 1, dim_feat))
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, num_joints, dim_feat))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                STMixerBlock(
                    dim=dim_feat,
                    dim_compress=dim_compress,
                    block_size=block_size,
                    block_stride=block_stride,
                    mlp_ratio=mlp_ratio,
                    mlp_out_ratio=mlp_out_ratio,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_scale=qkv_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    attn_head_dim=attn_head_dim,
                )
                for _ in range(depth)
            ]
        )

        self.init_std = init_std
        trunc_normal_(self.pos_embed_t, std=init_std)
        trunc_normal_(self.pos_embed_s, std=init_std)

    def forward(self, x):
        x = self.input_embed(x)
        x = x + self.pos_embed_t + self.pos_embed_s
        x = self.pos_drop(x)
        x = rearrange(x, "b t j c -> b c t j")
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "b c t j -> b t j c")
        return x


class STAPFormer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(
            depth=args.depth,
            dim_in=args.dim_in,
            dim_feat=args.dim_feat,
            dim_compress=args.dim_compress,
            block_size=args.block_size,
            block_stride=args.block_stride,
            mlp_ratio=args.mlp_ratio,
            mlp_out_ratio=args.mlp_out_ratio,
            num_heads=args.num_heads,
            qkv_bias=args.qkv_bias,
            qkv_scale=args.qkv_scale,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            attn_head_dim=args.attn_head_dim,
            init_std=args.init_std,
            num_frames=args.num_frames,
            num_joints=args.num_joints,
        )

        self.norm = nn.LayerNorm(args.dim_feat)
        self.rep_logit = nn.Sequential(
            nn.Linear(args.dim_feat, args.dim_rep), nn.Tanh()
        )
        self.head = nn.Linear(args.dim_rep, args.dim_out)

        self.init_std = args.init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_rep=False):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x
        x = self.head(x)
        return x


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.getcwd())

    from lib.utils.config import get_config
    from torchprofile import profile_macs

    config = get_config("configs/h36m/blockpose_small.yaml")
    print(config)
    model = BlockPose(config)

    model_params = sum(p.numel() for p in model.parameters())
    print("model_params:", model_params)

    inputs = torch.randn(1, config.num_frames, config.num_joints, 3)
    macs = profile_macs(model, inputs)
    print("macs:", macs)
