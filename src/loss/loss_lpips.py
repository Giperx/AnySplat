from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from src.dataset.types import BatchedExample
from src.misc.nn_module_tools import convert_to_buffer
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    conf: bool = False
    alpha: bool = False
    mask: bool = False


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        static_flag: bool = False, # 区分当前帧和历史帧
    ) -> Float[Tensor, ""]:
        image = (batch["context"]["image"] + 1) / 2
        using_index = batch.get("using_index")
        if not static_flag:
            using_index = using_index[:3]  # only first three views for current frame
        if using_index is not None:
            image = image[:, using_index]
        
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        
        mask = torch.ones_like(prediction.alpha, device=prediction.alpha.device).bool()  # 默认情况mask为全1
        if self.cfg.mask or self.cfg.alpha or self.cfg.conf:
            if self.cfg.mask:
                mask = batch["context"]["valid_mask"] # 全是-1, b v h w
                if using_index is not None:
                    mask = mask[:, using_index]
            elif self.cfg.alpha:
                mask = prediction.alpha
            elif self.cfg.conf:
                mask = depth_dict['conf_valid_mask']

        mask = mask.bool()

        if depth_dict is not None and 'dynamic_conf' in depth_dict and static_flag:
            dynamic_conf = depth_dict['dynamic_conf']
            if using_index is not None:
                dynamic_conf = dynamic_conf[:, using_index]
            static_mask = dynamic_conf < 0.5
            if static_mask.dim() == 4 and mask.dim() == 5:
                static_mask = static_mask.unsqueeze(2)
            elif static_mask.dim() == 5 and mask.dim() == 4:
                mask = mask.unsqueeze(2)
            mask = mask & static_mask

        total_views = prediction.color.shape[1]
        max_views = min(total_views, 3)

        if static_flag: # historical frame, only static part
            view_slice = slice(max_views, total_views)
        else: # current frame
            view_slice = slice(0, max_views)

        prediction_color = prediction.color[:, view_slice]
        image = image[:, view_slice]
        mask = mask[:, view_slice]

        if prediction_color.shape[1] == 0:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        if mask.dim() == 5:
            mask = mask.squeeze(2)

        b, v, c, h, w = prediction_color.shape
        expanded_mask = mask.unsqueeze(2).expand(-1, -1, c, -1, -1)
        masked_pred = prediction_color * expanded_mask
        masked_img = image * expanded_mask
        loss = self.lpips.forward(
            rearrange(masked_pred, "b v c h w -> (b v) c h w"),
            rearrange(masked_img, "b v c h w -> (b v) c h w"),
            normalize=True,
        )        
        
        return self.cfg.weight * torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)
