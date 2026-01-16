from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        static_flag: bool = False,
    ) -> Float[Tensor, ""]:
        # Get alpha and valid mask from inputs
        alpha = prediction.alpha
        # valid_mask = torch.ones_like(alpha, device=alpha.device).bool()
        valid_mask = batch['context']['valid_mask']
        using_index = batch.get("using_index")
        if not static_flag:
            using_index = using_index[:3]  # only first three views for current frame
        # # only for objaverse
        # if batch['context']['valid_mask'].sum() > 0:
        #     valid_mask = batch['context']['valid_mask']

        # Determine which mask to use based on config
        if self.cfg.mask:
            mask = valid_mask # 全是-1
        elif self.cfg.alpha:
            mask = alpha  
        elif self.cfg.conf:
            mask = depth_dict['conf_valid_mask']
        else:
            mask = torch.ones_like(alpha, device=alpha.device).bool() # 默认情况mask为全1

        ### get dynamic conf from depth_dict if exists
        if 'dynamic_conf' in depth_dict and static_flag == True:
            static_mask = depth_dict['dynamic_conf'] < 0.5
            ### check dimension
            if static_mask.dim() == 4 and mask.dim() == 5:
                static_mask = static_mask.unsqueeze(2)  # (B, V, 1, H, W)
            elif static_mask.dim() == 5 and mask.dim() == 4:
                mask = mask.unsqueeze(2)  # (B, V, 1, H, W)
            mask = mask & static_mask
            
        # Only use the first three context views to supervise the decoder
        max_views = min(prediction.color.shape[1], 3)
        

        # Rearrange and mask predicted and ground truth images
        if static_flag: # historical frame, only static part
            mask = mask[:, max_views:]
            pred_img = prediction.color[:, max_views:].permute(0, 1, 3, 4, 2)[mask]
            context_img = batch["context"]["image"][:, using_index[max_views:]]            
        else: # current frame
            mask = mask[:, :max_views]
            pred_img = prediction.color[:, :max_views].permute(0, 1, 3, 4, 2)[mask]
            context_img = batch["context"]["image"][:, using_index[:max_views]]
        gt_img = ((context_img + 1) / 2).permute(0, 1, 3, 4, 2)[mask]

        delta = pred_img - gt_img

        return self.cfg.weight * torch.nan_to_num((delta**2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
