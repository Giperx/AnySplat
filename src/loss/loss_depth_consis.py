from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss
from typing import Generic, Literal, Optional, TypeVar
from dataclasses import fields
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.loss.depth_anything.dpt import DepthAnything
from src.misc.utils import vis_depth_map

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossDepthConsisCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool
    loss_type: Literal['MSE', 'EdgeAwareLogL1', 'PearsonDepth'] = 'MSE'
    detach: bool = False
    conf: bool = False
    not_use_valid_mask: bool = False
    apply_after_step: int = 0

@dataclass
class LossDepthConsisCfgWrapper:
    depth_consis: LossDepthConsisCfg


class LogL1(torch.nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))

class EdgeAwareLogL1(torch.nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1, :]
        loss_y = lambda_y * logl1[..., :-1, :, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1, :]] = 0
                loss_y[~mask[..., :-1, :, :]] = 0
            return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1, :]]
            loss_y = loss_y[mask[..., :-1, :, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()
        
class LossDepthConsis(Loss[LossDepthConsisCfg, LossDepthConsisCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        
        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
        static_flag: bool = False, # 区分当前帧和历史帧
    ) -> Float[Tensor, ""]:
        
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0.0, dtype=torch.float32, device=prediction.depth.device)
        
        # Scale the depth between the near and far planes.
        using_index = batch.get("using_index")
        if not static_flag:
            using_index = using_index[:3]  # only first three views for current frame
            
        rendered_depth = prediction.depth
        gt_rgb = (batch["context"]["image"] + 1) / 2
        pred_depth = depth_dict['depth'].squeeze(-1)
        valid_mask = depth_dict["distill_infos"]['conf_mask'] # (B, V, H, W)

        if using_index is not None:
            rendered_depth = rendered_depth[:, using_index]
            gt_rgb = gt_rgb[:, using_index]
            pred_depth = pred_depth[:, using_index]
            valid_mask = valid_mask[:, using_index]

        if valid_mask.dim() == 5:
            valid_mask = valid_mask.squeeze(2)

        context_valid_mask = batch['context']['valid_mask']
        if using_index is not None:
            context_valid_mask = context_valid_mask[:, using_index]
        if context_valid_mask.dim() == 5:
            context_valid_mask = context_valid_mask.squeeze(2)
        context_valid_mask = context_valid_mask.bool()

        if context_valid_mask.sum() > 0:
            valid_mask = context_valid_mask

        valid_mask = valid_mask.bool()

        if depth_dict is not None and 'dynamic_conf' in depth_dict and static_flag:
            dynamic_conf = depth_dict['dynamic_conf']
            if using_index is not None:
                dynamic_conf = dynamic_conf[:, using_index]
            if dynamic_conf.dim() == 5:
                dynamic_conf = dynamic_conf.squeeze(2)
            static_mask = (dynamic_conf < 0.5).bool()
            valid_mask = valid_mask & static_mask

        total_views = rendered_depth.shape[1]
        max_views = min(total_views, 3)
        if static_flag:
            view_slice = slice(max_views, total_views)
        else:
            view_slice = slice(0, max_views)

        rendered_depth = rendered_depth[:, view_slice]
        pred_depth = pred_depth[:, view_slice]
        valid_mask = valid_mask[:, view_slice]
        gt_rgb = gt_rgb[:, view_slice]

        if self.cfg.not_use_valid_mask:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool, device=valid_mask.device)

        if rendered_depth.shape[1] == 0 or valid_mask.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=rendered_depth.device)

        pred_depth = pred_depth.detach() if self.cfg.detach else pred_depth
        if self.cfg.loss_type == 'MSE':
            depth_loss = F.mse_loss(rendered_depth, pred_depth, reduction='none')[valid_mask].mean()
        elif self.cfg.loss_type == 'EdgeAwareLogL1':
            rendered_depth = rendered_depth.flatten(0, 1).unsqueeze(-1)
            pred_depth = pred_depth.flatten(0, 1).unsqueeze(-1)
            gt_rgb = gt_rgb.flatten(0, 1).permute(0, 2, 3, 1)
            valid_mask = valid_mask.flatten(0, 1).unsqueeze(-1)
            depth_loss = EdgeAwareLogL1()(rendered_depth, pred_depth, gt_rgb, valid_mask)
        return self.cfg.weight * torch.nan_to_num(depth_loss, nan=0.0)