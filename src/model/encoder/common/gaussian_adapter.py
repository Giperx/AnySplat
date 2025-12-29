from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from src.geometry.projection import get_world_rays
from src.misc.sh_rotation import rotate_sh
from .gaussians import build_covariance

from ...types import Gaussians

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
        
    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        # levels: Float[Tensor, "*#batch"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)
        
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        # print(scales.max())
        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means.float(),
            # levels=levels.int(),
            covariances=covariances.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )
        
class UnifiedGaussianAdapterForDGGT(GaussianAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 假设 sh_degree 在初始化时已知，如果只是RGB，则 d_sh * 3 = 3 (degree 0)
        # 如果 self.d_sh 是每个颜色的系数数量 (比如 (degree+1)**2)
        # 根据你提供的 output_dim = 3 + 1 + 3 + 4 + 1，这里的 Color 是 3 通道
        pass

    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"], # 这是外部计算好传入的最终 opacity
        raw_gaussians: Float[Tensor, "*#batch _"], # 这是 Voxelize 后的特征
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        # 定义各部分通道长度
        # 根据 output_dim = 3(Color) + 1(Opacity) + 3(Scale) + 4(Rot) (+1 Conf outside)
        c_color = 3  # 如果是高阶SH，这里需要改为 ((sh_degree + 1)**2) * 3
        c_opa = 1
        c_scale = 3
        c_rot = 4
        
        # 按照 gs_activate_head 的逻辑进行 Split
        # 此时 raw_gaussians 不包含 Confidence (在外面已经被剥离)
        # 顺序: [Color, Opacity, Scale, Rotation]
        color, _, scales, rotations = torch.split(
            raw_gaussians, 
            [c_color, c_opa, c_scale, c_rot], 
            dim=-1
        )
        
        # --- 1. Scale Activation ---
        # 对应 gs_activate_head: scale = 0.1 * F.softplus(scale)
        scales = 0.1 * F.softplus(scales)
        
        # --- 2. Rotation Activation ---
        # 对应 gs_activate_head: rotation = F.normalize(rotation, dim=-1)
        rotations = F.normalize(rotations, dim=-1)
        
        # --- 3. Color / SH Processing ---
        # 对应 gs_activate_head: if sh_degree is None: color = torch.sigmoid(color)
        # 注意：这里 opacities 是外部传入的，所以不需要从 raw_gaussians 取出的 opacity
        
        # 如果是纯 RGB (dim=3)
        if c_color == 3:
            sh = torch.sigmoid(color) # 限制在 [0, 1]
            # 如果 Gaussian 类需要 SH 格式，可能需要 unsqueeze，视你的 Gaussians 类定义而定
            # 假设 Gaussians 类可以直接接受 RGB 作为 harmonics 的 0阶项
            # 或者我们需要把它 reshape 成 sh 格式
            sh = sh.unsqueeze(-2) # [..., 1, 3] -> 1个基函数, 3个颜色通道
        else:
            # 如果是高阶 SH，通常不加 sigmoid，直接作为系数
            sh = rearrange(color, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            # sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask # 如果需要 mask
        
        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means.float(),
            covariances=covariances.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )

class Unet3dGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)
        
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

