import copy

# VGGT parts
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from src.dataset.shims.bounds_shim import apply_bounds_shim
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.types import BatchedExample, DataShim
from src.geometry.projection import sample_image_grid

from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.encoder.heads.GaussianHead import GaussianHead, DPTHeadDGGT
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.geometry import get_rel_pos  # used for model hub
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max

from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone

from .backbone.croco.misc import transpose_to_landscape
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder, EncoderOutput
from .heads import head_factory
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

root_path = os.path.abspath(".")
sys.path.append(root_path)
from src.model.encoder.heads.head_modules import TransformerBlockSelfAttn
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from src.model.encoder.vggt.layers.mlp import Mlp
from src.model.encoder.vggt.models.vggt import VGGT

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pred_pose: bool = True
    frozenAggregator: bool = False
    frozenGaussianHead: bool = False
    frozenCameraHead: bool = False
    frozenDepthHead: bool = False
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = False
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__(cfg)
        # model_full = VGGT.from_pretrained("facebook/VGGT-1B")
        model_full = VGGT()
        self.aggregator = model_full.aggregator.to(torch.bfloat16)
        self.freeze_backbone = cfg.freeze_backbone
        self.distill = cfg.distill
        self.frozenAggregator = cfg.frozenAggregator
        self.frozenGaussianHead = cfg.frozenGaussianHead
        self.frozenCameraHead = cfg.frozenCameraHead
        self.frozenDepthHead = cfg.frozenDepthHead
        self.pred_pose = cfg.pred_pose

        self.camera_head = model_full.camera_head
        if self.cfg.pred_head_type == "depth":
            self.depth_head = model_full.depth_head
        else:
            self.point_head = model_full.point_head

        ### delete
        # if self.distill: 
        #     self.distill_aggregator = copy.deepcopy(self.aggregator)
        #     self.distill_camera_head = copy.deepcopy(self.camera_head)
        #     self.distill_depth_head = copy.deepcopy(self.depth_head)
        #     for module in [
        #         self.distill_aggregator,
        #         self.distill_camera_head,
        #         self.distill_depth_head,
        #     ]:
        #         for param in module.parameters():
        #             param.requires_grad = False
        #             param.data = param.data.cpu()

        if self.freeze_backbone:
            # Freeze backbone components
            if self.cfg.pred_head_type == "depth":
                for module in [self.aggregator, self.camera_head, self.depth_head]:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.aggregator, self.camera_head, self.point_head]:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # aggregator freeze
            freeze_module = self.cfg.freeze_module
            if freeze_module == "None":
                pass

            elif freeze_module == "all":
                for param in self.aggregator.parameters():
                    param.requires_grad = False

            else:
                module_pairs = {
                    "patch_embed+frame": ["patch_embed", "frame"],
                    "patch_embed+global": ["patch_embed", "global"],
                    "global+frame": ["global", "frame"],
                }

                if freeze_module in module_pairs:
                    for name, param in self.aggregator.named_parameters():
                        if any(m in name for m in module_pairs[freeze_module]):
                            param.requires_grad = False
                else:
                    for name, param in self.named_parameters():
                        param.requires_grad = (
                            freeze_module not in name and "distill" not in name
                        )

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity, 1+7+3
        self.voxel_size = cfg.voxel_size
        self.gs_params_head_type = cfg.gs_params_head_type
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            output_dim=self.raw_gs_dim + 1, # 1 for confidence
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
        )

        ### from DGGT
        # self.gs_head = GaussianHead(dim_in= 3 * head_params.enc_embed_dim, output_dim=3 + 1 + 3 + 4 + 1, activation="sigmoid", feature_only=True)# ,down_ratio=2)#RGB
        self.dynamic_head = DPTHeadDGGT(dim_in= 1024, output_dim = 1 + 1, activation="linear") # ,down_ratio=2)#RGB
        
        print("self.frozenAggregator:", self.frozenAggregator,
              "self.frozenGaussianHead:", self.frozenGaussianHead,
              "self.frozenCameraHead:", self.frozenCameraHead,
              "self.frozenDepthHead:", self.frozenDepthHead,
        )
        ### Freeze specific components
        if self.frozenAggregator:
            for param in self.aggregator.parameters():
                param.requires_grad = False
        if self.frozenGaussianHead:
            for param in self.gaussian_param_head.parameters():
                param.requires_grad = False
        if self.frozenCameraHead:
            for param in self.camera_head.parameters():
                param.requires_grad = False
        if self.frozenDepthHead:
            for param in self.depth_head.parameters():
                param.requires_grad = False


    ### add pretrained weights loading pretrain AnySplat model
    def usePreTrainedWeights(self):
        print("self.frozenAggregator:", self.frozenAggregator,
              "self.frozenGaussianHead:", self.frozenGaussianHead,
              "self.frozenCameraHead:", self.frozenCameraHead,
              "self.frozenDepthHead:", self.frozenDepthHead,
        )
        if self.frozenAggregator:
            for param in self.aggregator.parameters():
                param.requires_grad = False
        if self.frozenGaussianHead:
            for param in self.gaussian_param_head.parameters():
                param.requires_grad = False
        if self.frozenCameraHead:
            for param in self.camera_head.parameters():
                param.requires_grad = False
        if self.frozenDepthHead:
            for param in self.depth_head.parameters():
                param.requires_grad = False

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_pts3d(self, pts3ds, valid_masks, original_extrinsics=None):
        # normalize pts_all
        B = pts3ds.shape[0]
        pts3d_norms = []
        scale_factors = []
        for bs in range(B):
            pts3d, valid_mask = pts3ds[bs], valid_masks[bs]
            if original_extrinsics is not None:
                camera_c2w = original_extrinsics[bs]
                first_camera_w2c = (
                    camera_c2w[0].inverse().unsqueeze(0).repeat(pts3d.shape[0], 1, 1)
                )

                pts3d_homo = torch.cat(
                    [pts3d, torch.ones_like(pts3d[:, :, :, :1])], dim=-1
                )
                transformed_pts3d = torch.bmm(
                    first_camera_w2c, pts3d_homo.flatten(1, 2).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                scene_scale = torch.norm(
                    transformed_pts3d.flatten(0, 1)[valid_mask.flatten(0, 2).bool()],
                    dim=-1,
                ).mean()
            else:
                transformed_pts3d = pts3d[valid_mask]
                dis = transformed_pts3d.norm(dim=-1)
                scene_scale = dis.mean().clip(min=1e-8)
            # pts3d_norm[bs] = pts3d[bs] / scene_scale
            pts3d_norms.append(pts3d / scene_scale)
            scale_factors.append(scene_scale)
        return torch.stack(pts3d_norms, dim=0), torch.stack(scale_factors, dim=0)

    def align_pts_all_with_pts3d(
        self, pts_all, pts3d, valid_mask, original_extrinsics=None
    ):
        # align pts_all with pts3d
        B = pts_all.shape[0]

        # follow vggt's normalization implementation
        pts3d_norm, scale_factor = self.normalize_pts3d(
            pts3d, valid_mask, original_extrinsics
        )  # check if this is correct
        pts_all = pts_all * scale_factor.view(B, 1, 1, 1, 1)

        return pts_all

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = image.device
        b, v, _, h, w = image.shape
        distill_infos = {}
        # if self.distill: ### delete
        #     distill_image = image.clone().detach()
            
        #     for module in [
        #         self.distill_aggregator,
        #         self.distill_camera_head,
        #         self.distill_depth_head,
        #     ]:
        #         for param in module.parameters():
        #             param.data = param.data.to(device, non_blocking=True)

        #     with torch.no_grad(): 
        #         # Process with bfloat16 precision
        #         with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        #             distill_aggregated_tokens_list, distill_patch_start_idx = (
        #                 self.distill_aggregator(
        #                     distill_image.to(torch.bfloat16),
        #                     intermediate_layer_idx=self.cfg.intermediate_layer_idx,
        #                 )
        #             )

        #         # Process with default precision
        #         with torch.amp.autocast("cuda", enabled=False):
        #             # Get camera pose information
        #             distill_pred_pose_enc_list = self.distill_camera_head(
        #                 distill_aggregated_tokens_list
        #             )
        #             last_distill_pred_pose_enc = distill_pred_pose_enc_list[-1]
        #             distill_extrinsic, distill_intrinsic = pose_encoding_to_extri_intri(
        #                 last_distill_pred_pose_enc, image.shape[-2:]
        #             )

        #             # Get depth information
        #             distill_depth_map, distill_depth_conf = self.distill_depth_head(
        #                 distill_aggregated_tokens_list,
        #                 images=distill_image,
        #                 patch_start_idx=distill_patch_start_idx,
        #             )

        #             # Convert depth to 3D points
        #             distill_pts_all = batchify_unproject_depth_map_to_point_map(
        #                 distill_depth_map, distill_extrinsic, distill_intrinsic
        #             )
        #         # Store results
        #         distill_infos["pred_pose_enc_list"] = distill_pred_pose_enc_list
        #         distill_infos["pts_all"] = distill_pts_all
        #         distill_infos["depth_map"] = distill_depth_map

        #         conf_threshold = torch.quantile(
        #             distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True
        #         )  # Get threshold for each view
        #         conf_mask = distill_depth_conf > conf_threshold.unsqueeze(-1)
        #         distill_infos["conf_mask"] = conf_mask

        #         for module in [
        #             self.distill_aggregator,
        #             self.distill_camera_head,
        #             self.distill_depth_head,
        #         ]:
        #             for param in module.parameters():
        #                 param.data = param.data.cpu()
        #         # Clean up to save memory
        #         del distill_aggregated_tokens_list, distill_patch_start_idx
        #         del distill_pred_pose_enc_list, last_distill_pred_pose_enc
        #         del distill_extrinsic, distill_intrinsic
        #         del distill_depth_map, distill_depth_conf
        #         torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            # aggregated_tokens_list, patch_start_idx = self.aggregator(
            #     image.to(torch.bfloat16),
            #     intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            # )
            
            ### for DGGT aggregator
            aggregated_tokens_list, image_tokens_list, dino_token_list, patch_start_idx = self.aggregator(
                image.to(torch.bfloat16),
                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            )
            
        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug

            if self.cfg.pred_head_type == "point":
                pts_all, pts_conf = self.point_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
            elif self.cfg.pred_head_type == "depth":
                depth_map, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )
            else:
                raise ValueError(f"Invalid pred_head_type: {self.cfg.pred_head_type}")

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)
                
            ### add conf_mask for loss_depth_consis
            depth_conf_detached = depth_conf.detach()
            conf_threshold_tmp = torch.quantile(
                depth_conf_detached.flatten(2, 3), 0.3, dim=-1, keepdim=True
            )  # Get threshold for each view
            conf_mask_tmp = depth_conf > conf_threshold_tmp.unsqueeze(-1)
            distill_infos["conf_mask"] = conf_mask_tmp
            del depth_conf_detached, conf_threshold_tmp, conf_mask_tmp

        # dpt style gs_head input format
        out = self.gaussian_param_head(
            aggregated_tokens_list,
            pts_all.flatten(0, 1).permute(0, 3, 1, 2),
            image,
            patch_start_idx=patch_start_idx,
            image_size=(h, w),
        )
        
        ### infer gs_head like DGGT
        # out = self.gs_head(image_tokens_list, image, patch_start_idx)
        # dynamic_conf: (B, V, H, W, 1) depending on implementation
        dynamic_conf, _ = self.dynamic_head(dino_token_list, image, patch_start_idx)
        # print("*******dynamic_conf shape:", dynamic_conf.shape) # torch.Size([1, 6, 224, 224, 1])
        
        del aggregated_tokens_list, patch_start_idx, image_tokens_list, dino_token_list
        torch.cuda.empty_cache()

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        # 确保 dynamic_conf 维度正确 (B, V, H, W)
        if dynamic_conf.dim() == 4 and dynamic_conf.shape[0] == b * v:
             dynamic_conf = rearrange(dynamic_conf, "(b v) 1 h w -> b v h w", b=b, v=v)
        elif dynamic_conf.dim() == 5: # run in here line
             dynamic_conf = dynamic_conf.squeeze(4) # (B, V, H, W)
             
        infos = {}
        # infos["dynamic_mask"] = dynamic_conf > 0.5  # Assuming > 0.5 is dynamic based on your description   
        

        # 同样处理 out (Feature & Conf)
        if out.dim() == 4 and out.shape[0] == b * v:
             out = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
        
        anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        # 引入 torch_scatter 用于内联的 voxelization
        # from torch_scatter import scatter_max, scatter_add

        views_per_step = 3 # 假设前3帧是当前时刻(t)，后3帧是上一时刻(t-1)
        
        neural_feats_list, neural_pts_list = [], []
        
        # Init lists for static-only gaussians
        static_neural_feats_list = []
        static_neural_pts_list = []

        if self.cfg.voxelize:
            for b_i in range(b):
                # --- 1. 当前时刻 (Current Timestep): 保留所有内容 (Dynamic + Static) ---
                curr_feats = anchor_feats[b_i, :views_per_step] # (V_t, C, H, W)
                curr_pts = pts_all[b_i, :views_per_step]        # (V_t, H, W, 3)
                curr_conf = conf[b_i, :views_per_step]          # (V_t, H, W)
                # Static check for current
                curr_d_conf = dynamic_conf[b_i, :views_per_step]

                # Flatten
                curr_feats_flat = rearrange(curr_feats, "v c h w -> (v h w) c")
                curr_pts_flat = rearrange(curr_pts, "v h w c -> (v h w) c")
                curr_conf_flat = rearrange(curr_conf, "v h w -> (v h w)")
                curr_d_conf_flat = rearrange(curr_d_conf, "v h w -> (v h w)")

                # --- 2. 其他/上一时刻 (Other Timestep): 仅保留静态内容 (conf < 0.5) ---
                prev_feats = anchor_feats[b_i, views_per_step:]
                prev_pts = pts_all[b_i, views_per_step:]
                prev_conf = conf[b_i, views_per_step:]
                prev_d_conf = dynamic_conf[b_i, views_per_step:] # (V_prev, H, W)

                # 筛选静态部分: dynamic_conf < 0.5
                static_mask = prev_d_conf < 0.5
                static_mask_flat = rearrange(static_mask, "v h w -> (v h w)")

                prev_feats_flat = rearrange(prev_feats, "v c h w -> (v h w) c")
                prev_pts_flat = rearrange(prev_pts, "v h w c -> (v h w) c")
                prev_conf_flat = rearrange(prev_conf, "v h w -> (v h w)")

                # 应用 Mask (Prev is always static-filtered in original logic)
                prev_feats_static = prev_feats_flat[static_mask_flat]
                prev_pts_static = prev_pts_flat[static_mask_flat]
                prev_conf_static = prev_conf_flat[static_mask_flat]

                # --- 3. 合并数据 (Merge) for Main Gaussians ---
                merged_feats = torch.cat([curr_feats_flat, prev_feats_static], dim=0) # (N_total, C)
                merged_pts = torch.cat([curr_pts_flat, prev_pts_static], dim=0)       # (N_total, 3)
                merged_conf = torch.cat([curr_conf_flat, prev_conf_static], dim=0)    # (N_total)

                # --- 3b. 合并数据 (Merge) for Static Gaussians ---
                # For static gaussians, we filter current frame as well
                curr_static_mask = curr_d_conf_flat < 0.5
                curr_feats_static = curr_feats_flat[curr_static_mask]
                curr_pts_static = curr_pts_flat[curr_static_mask]
                curr_conf_static = curr_conf_flat[curr_static_mask]

                merged_feats_static_only = torch.cat([curr_feats_static, prev_feats_static], dim=0)
                merged_pts_static_only = torch.cat([curr_pts_static, prev_pts_static], dim=0)
                merged_conf_static_only = torch.cat([curr_conf_static, prev_conf_static], dim=0)

                # --- 4. 内联 Voxelization (Main) ---
                def process_voxelization(m_pts, m_feats, m_conf):
                    if m_pts.shape[0] == 0:
                        return m_feats, m_pts # Handle empty case if needed
                        
                    v_indices = (m_pts / self.voxel_size).round().int()
                    _, inv_indices = torch.unique(v_indices, dim=0, return_inverse=True)
                    
                    # 依据 Confidence 进行加权融合
                    c_voxel_max, _ = scatter_max(m_conf, inv_indices, dim=0)
                    c_exp = torch.exp(m_conf - c_voxel_max[inv_indices])
                    v_weights = scatter_add(c_exp, inv_indices, dim=0)
                    w = (c_exp / (v_weights[inv_indices] + 1e-6)).unsqueeze(-1)

                    w_pts = m_pts * w
                    w_feats = m_feats * w

                    v_pts = scatter_add(w_pts, inv_indices, dim=0)
                    v_feats = scatter_add(w_feats, inv_indices, dim=0)
                    return v_feats, v_pts

                voxel_feats, voxel_pts = process_voxelization(merged_pts, merged_feats, merged_conf)
                neural_feats_list.append(voxel_feats)
                neural_pts_list.append(voxel_pts)

                # --- 4b. 内联 Voxelization (Static) ---
                s_voxel_feats, s_voxel_pts = process_voxelization(merged_pts_static_only, merged_feats_static_only, merged_conf_static_only)
                static_neural_feats_list.append(s_voxel_feats)
                static_neural_pts_list.append(s_voxel_pts)

        else:
            # 非 Voxelize 模式 (Simple Concatenation with Mask)
            # 确保 dynamic_conf 维度正确 (B, V, H, W)
            if dynamic_conf.shape[0] == b * v:
                 d_conf_reshaped = rearrange(dynamic_conf, "(b v) 1 h w -> b v h w", b=b, v=v)
            else:
                 d_conf_reshaped = dynamic_conf.squeeze(2)

            for b_i in range(b):
                # 1. 基础几何有效性掩码 (来自 depth unproject 等)
                valid_mask = conf_valid_mask[b_i]  # [V, H, W]
                
                # 2. 构建时序动静分离掩码
                # 初始化为 True，默认保留所有点
                temporal_mask = torch.ones_like(valid_mask, dtype=torch.bool)
                
                # 对历史时刻 (views_per_step 之后的帧)，只保留静态部分 (conf < 0.5)
                # 假设 views_per_step = 3，即前3帧为当前时刻，后3帧为历史时刻
                if v > views_per_step:
                    is_static = d_conf_reshaped[b_i, views_per_step:] < 0.5
                    temporal_mask[views_per_step:] = is_static
                
                # 3. 合并掩码: 既要几何有效，又要符合动静保留策略
                final_mask = valid_mask & temporal_mask
                
                # 4. 提取特征和点云
                neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[final_mask]
                )
                neural_pts_list.append(pts_all[b_i][final_mask])

                # --- Static Only Processing ---
                # For static list, we require ALL frames to be static
                # is_static_all: [V, H, W]
                is_static_all = d_conf_reshaped[b_i] < 0.5
                final_static_mask = valid_mask & is_static_all
                
                static_neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[final_static_mask]
                )
                static_neural_pts_list.append(pts_all[b_i][final_static_mask])


        # ### ---------- 处理为函数process_gaussian_list ----------
        # max_voxels = max(f.shape[0] for f in neural_feats_list)
        # neural_feats = self.pad_tensor_list(
        #     neural_feats_list, (max_voxels,), value=-1e10
        # )

        # neural_pts = self.pad_tensor_list(
        #     neural_pts_list, (max_voxels,), -1e4
        # )  # -1 == invalid voxel

        # depths = neural_pts[..., -1].unsqueeze(-1)
        # densities = neural_feats[..., 0].sigmoid()

        # ### for DGGT
        # # opacity_idx = 3
        # # densities = neural_feats[..., opacity_idx].sigmoid()
        
        # assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        # assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        # opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        # if self.cfg.opacity_conf:
        #     shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
        #     opacity = opacity * torch.sigmoid(depth_conf - shift)[
        #         conf_valid_mask
        #     ].unsqueeze(
        #         0
        #     )  # little bit hacky
        # # GS Prune, but only works when bs = 1
        # # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # # Note: we not prune gaussians here, but we will try it in the future
        # if self.cfg.gs_prune and b == 1:
        #     opacity_threshold = self.cfg.opacity_threshold
        #     gaussian_usage = opacity > opacity_threshold  # (B, N)

        #     print(
        #         f"based on opacity threshold {opacity_threshold}, pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
        #     )

        #     if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
        #         # rank by opacity
        #         num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
        #         idx_sort = opacity.argsort(dim=1, descending=True)
        #         keep_idx = idx_sort[:, :num_keep]
        #         gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
        #         gaussian_usage.scatter_(1, keep_idx, True)

        #     neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
        #     depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
        #     neural_feats = (
        #         neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
        #     )
        #     opacity = opacity[gaussian_usage].view(b, -1).contiguous()

        #     print(
        #         f"finally pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
        #     )
        # gaussians = self.gaussian_adapter.forward(
        #     neural_pts,
        #     depths,
        #     opacity,
        #     neural_feats[..., 1:].squeeze(2),
        # )
        # ### for DGGT
        # # gaussians = self.gaussian_adapter.forward(
        # #     neural_pts,
        # #     depths,
        # #     opacity,
        # #     neural_feats, # 传入完整特征 [Color, Opacity, Scale, Rot]
        # # )
        # ### -------------
        
        # === Helper function to process gaussians (avoid code duplication) ===
        def process_gaussian_list(n_feats_list, n_pts_list, tag="main"):
            max_v = max(f.shape[0] for f in n_feats_list)
            n_feats = self.pad_tensor_list(n_feats_list, (max_v,), value=-1e10)
            n_pts = self.pad_tensor_list(n_pts_list, (max_v,), -1e4)

            d = n_pts[..., -1].unsqueeze(-1)
            dens = n_feats[..., 0].sigmoid()
            ### for DGGT
            # opacity_idx = 3
            # densities = neural_feats[..., opacity_idx].sigmoid()

            op = self.map_pdf_to_opacity(dens, global_step).squeeze(-1)
            if self.cfg.opacity_conf:
                shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
                op = op * torch.sigmoid(depth_conf - shift)[conf_valid_mask].unsqueeze(0)

            # GS Prune (Only for bs=1 per original logic)
            if self.cfg.gs_prune and b == 1:
                opacity_threshold = self.cfg.opacity_threshold
                gaussian_usage = op > opacity_threshold
                
                # Logic for keep ratio
                if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                    num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                    idx_sort = op.argsort(dim=1, descending=True)
                    keep_idx = idx_sort[:, :num_keep]
                    gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                    gaussian_usage.scatter_(1, keep_idx, True)
                
                n_pts = n_pts[gaussian_usage].view(b, -1, 3).contiguous()
                d = d[gaussian_usage].view(b, -1, 1).contiguous()
                n_feats = n_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
                op = op[gaussian_usage].view(b, -1).contiguous()
                
                if tag == "main":
                    print(f"[{tag}] pruned {gaussian_usage.shape[1] - n_pts.shape[1]} gaussians")

            gs = self.gaussian_adapter.forward(
                n_pts, d, op, n_feats[..., 1:].squeeze(2),
            )
            ### for DGGT
            # gs = self.gaussian_adapter.forward(
            #     n_pts,
            #     d,
            #     op,
            #     n_feats, # 传入完整特征 [Color, Opacity, Scale, Rot]
            # )
            return gs, dens, n_pts
        
        # 1. Process Main Gaussians
        gaussians, densities, neural_pts = process_gaussian_list(neural_feats_list, neural_pts_list, tag="main")
        
        # 2. Process Static Gaussians
        static_gaussians, _, _ = process_gaussian_list(static_neural_feats_list, static_neural_pts_list, tag="static")

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                pts_all[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                "b v (h w) srf s -> b v h w srf s",
                h=h,
                w=w,
            )

        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)

        if global_step < 10 or global_step % 300 == 0:
            print(
                f"scene scale: {scene_scale:.3f}, pixel-wise num: {h*w*v}, after voxelize: {neural_pts.shape[1]}, voxelize ratio: {infos['voxelize_ratio']:.3f}"
            )
            print(
                f"Gaussians attributes: \n"
                f"opacities: mean: {gaussians.opacities.mean()}, min: {gaussians.opacities.min()}, max: {gaussians.opacities.max()} \n"
                f"scales: mean: {gaussians.scales.mean()}, min: {gaussians.scales.min()}, max: {gaussians.scales.max()}"
            )

            print("B:", b, "V:", v, "H:", h, "W:", w)
        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic = intrinsic.clone()  # Create a new tensor
        intrinsic = torch.stack(
            [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
        )


        return EncoderOutput(
            gaussians=gaussians, # 当前帧的动静高斯+历史帧静态高斯
            static_gaussians=static_gaussians, # 历史帧静态高斯
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic,
            ),
            depth_dict=dict(depth=depth_map, conf_valid_mask=conf_valid_mask, dynamic_conf=dynamic_conf), # 增加dynamic_conf
            infos=infos,
            distill_infos=distill_infos,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
