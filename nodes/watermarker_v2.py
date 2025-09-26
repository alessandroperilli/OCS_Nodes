import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class OCS_WatermarkerV2:
    """Overlay a watermark onto the bottom-right corner of an image using the
    same tensor compositing pattern as the LayerSystem node so partially
    transparent edges stay intact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "watermark": ("IMAGE",),
                "scale_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "padding": ("INT", {"default": 25, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("watermarked_image",)
    FUNCTION = "apply_watermark"
    CATEGORY = "OCS Nodes"

    # ---------------------------------------------------------------------
    def apply_watermark(self, source_image, watermark, scale_percent, padding):
        padding = self._extract_scalar(padding, int)
        scale_percent = self._extract_scalar(scale_percent, float)

        src_batch = self._ensure_tensor(source_image)
        wm_batch = self._ensure_tensor(watermark)

        batch_size = src_batch.shape[0]
        wm_count = wm_batch.shape[0]

        outputs = []
        for idx in range(batch_size):
            src_img = src_batch[idx]
            wm_img = wm_batch[idx % wm_count]
            outputs.append(
                self._apply_single(src_img, wm_img, scale_percent, padding)
            )

        result = torch.stack(outputs, dim=0).to(device=src_batch.device)
        return (result,)

    # ------------------------------------------------------------------
    def _apply_single(
        self,
        src_img: torch.Tensor,
        wm_img: torch.Tensor,
        scale_percent: float,
        padding: int,
    ) -> torch.Tensor:
        src = src_img.to(dtype=torch.float32)
        wm = wm_img.to(device=src.device, dtype=torch.float32)

        src_rgb, src_alpha, src_extra = self._split_source(src)
        wm_rgb, wm_alpha = self._split_watermark(wm)

        src_h, src_w = src_rgb.shape[0], src_rgb.shape[1]
        wm_h, wm_w = wm_rgb.shape[0], wm_rgb.shape[1]

        if scale_percent <= 0.0 or wm_h == 0 or wm_w == 0:
            return self._reassemble(src_rgb, src_alpha, src_extra, src_img)

        max_width = max(1, src_w - padding * 2)
        max_height = max(1, src_h - padding * 2)

        target_w = max(1, int(round(src_w * max(scale_percent, 0.0) / 100.0)))
        target_h = max(1, int(round(src_h * max(scale_percent, 0.0) / 100.0)))

        target_w = min(target_w, max_width)
        target_h = min(target_h, max_height)

        resize_ratio = min(target_w / wm_w, target_h / wm_h)
        resize_ratio = max(resize_ratio, 1e-8)

        new_w = max(1, int(round(wm_w * resize_ratio)))
        new_h = max(1, int(round(wm_h * resize_ratio)))

        wm_rgb_resized = self._resize_tensor(wm_rgb, new_h, new_w)
        wm_alpha_resized = self._resize_tensor(wm_alpha, new_h, new_w)

        wm_layer_rgb = torch.zeros(
            (src_h, src_w, 3), device=src_rgb.device, dtype=src_rgb.dtype
        )
        wm_layer_alpha = torch.zeros(
            (src_h, src_w, 1), device=src_rgb.device, dtype=src_rgb.dtype
        )

        x_pos = self._anchor_position(src_w, new_w, padding)
        y_pos = self._anchor_position(src_h, new_h, padding)

        wm_layer_rgb[y_pos : y_pos + new_h, x_pos : x_pos + new_w, :] = wm_rgb_resized
        wm_layer_alpha[y_pos : y_pos + new_h, x_pos : x_pos + new_w, :] = (
            wm_alpha_resized.clamp(0.0, 1.0)
        )

        wm_layer_alpha = wm_layer_alpha.clamp(0.0, 1.0)

        if src_alpha is None:
            composed_rgb = src_rgb * (1.0 - wm_layer_alpha) + wm_layer_rgb * wm_layer_alpha
            composed_alpha = None
        else:
            base_alpha = src_alpha.clamp(0.0, 1.0)
            top_alpha = wm_layer_alpha

            base_premult = src_rgb * base_alpha
            top_premult = wm_layer_rgb * top_alpha

            out_alpha = top_alpha + base_alpha * (1.0 - top_alpha)

            safe_alpha = out_alpha.clamp_min(1e-6)
            out_rgb_premult = top_premult + base_premult * (1.0 - top_alpha)
            composed_rgb = torch.where(
                out_alpha > 1e-6,
                out_rgb_premult / safe_alpha,
                torch.zeros_like(out_rgb_premult),
            )
            composed_alpha = out_alpha

        return self._reassemble(composed_rgb, composed_alpha, src_extra, src_img)

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_tensor(img):
        if isinstance(img, list):
            img = img[0]
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if not torch.is_floating_point(img):
            img = img.float()
        return img

    @staticmethod
    def _extract_scalar(value, caster):
        if isinstance(value, list):
            value = value[0]
        return caster(value)

    @staticmethod
    def _split_source(
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        channels = tensor.shape[-1]
        if channels == 1:
            rgb = tensor.repeat(1, 1, 3)
            alpha = None
            extra = None
        elif channels == 2:
            rgb = tensor[..., :1].repeat(1, 1, 3)
            alpha = tensor[..., 1:2]
            extra = None
        elif channels == 3:
            rgb = tensor[..., :3]
            alpha = None
            extra = None
        else:
            rgb = tensor[..., :3]
            alpha = tensor[..., 3:4]
            extra = tensor[..., 4:] if channels > 4 else None
        return rgb, alpha, extra

    @staticmethod
    def _split_watermark(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = tensor.shape[-1]
        if channels >= 4:
            rgb = tensor[..., :3]
            alpha = tensor[..., 3:4]
        elif channels == 3:
            rgb = tensor
            alpha = torch.ones_like(rgb[..., :1])
        elif channels == 1:
            rgb = tensor.repeat(1, 1, 3)
            alpha = torch.ones_like(rgb[..., :1])
        else:
            raise ValueError("Unsupported number of channels in watermark image")
        return rgb, alpha

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
        if tensor.shape[0] == new_h and tensor.shape[1] == new_w:
            return tensor
        data = tensor.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(data, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    @staticmethod
    def _anchor_position(base: int, layer: int, padding: int) -> int:
        max_start = max(0, base - layer)
        desired = base - layer - padding
        return max(0, min(max_start, desired))

    @staticmethod
    def _reassemble(
        rgb: torch.Tensor,
        alpha: Optional[torch.Tensor],
        extra: Optional[torch.Tensor],
        template: torch.Tensor,
    ) -> torch.Tensor:
        result = rgb
        if alpha is not None:
            result = torch.cat([result, alpha], dim=-1)
        if extra is not None:
            result = torch.cat([result, extra], dim=-1)
        return result.clamp(0.0, 1.0).to(dtype=template.dtype, device=template.device)


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}
