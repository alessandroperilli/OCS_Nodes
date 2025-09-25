import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class OCS_WatermarkerV2:
    """Overlay a watermark onto the bottom-right corner of an image while
    preserving transparency by following the same tensor-based compositing
    strategy used in the LayerSystem node."""

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

    # ------------------------------------------------------------------
    def apply_watermark(self, source_image, watermark, scale_percent, padding):
        padding = self._extract_scalar(padding, int)
        scale_percent = self._extract_scalar(scale_percent, float)

        src_tensor = self._ensure_tensor(source_image)
        wm_tensor = self._ensure_tensor(watermark)

        batch_size = src_tensor.shape[0]
        wm_count = wm_tensor.shape[0]

        results = []
        for idx in range(batch_size):
            src_img = src_tensor[idx]
            wm_img = wm_tensor[idx % wm_count]
            overlaid = self._overlay_watermark(src_img, wm_img, scale_percent, padding)
            results.append(overlaid)

        stacked = torch.stack(results, dim=0).to(device=src_tensor.device)
        return (stacked,)

    # ------------------------------------------------------------------
    def _overlay_watermark(
        self,
        src_img_tensor: torch.Tensor,
        wm_tensor: torch.Tensor,
        scale_percent: float,
        padding: int,
    ) -> torch.Tensor:
        src = src_img_tensor.to(dtype=torch.float32)
        wm = wm_tensor.to(dtype=torch.float32, device=src_img_tensor.device)

        src_rgb, src_alpha, src_extra = self._split_channels(src)
        src_rgb = src_rgb.clone()
        if src_alpha is not None:
            src_alpha = src_alpha.clone()
        if src_extra is not None:
            src_extra = src_extra.clone()

        wm_rgba = self._ensure_rgba(wm)

        src_h, src_w = src_rgb.shape[0], src_rgb.shape[1]
        wm_h, wm_w = wm_rgba.shape[0], wm_rgba.shape[1]

        if scale_percent <= 0.0 or wm_h == 0 or wm_w == 0:
            return self._reassemble(src_rgb, src_alpha, src_extra, src_img_tensor)

        base_available_w = max(1, src_w - padding * 2)
        base_available_h = max(1, src_h - padding * 2)

        requested_scale = max(scale_percent / 100.0, 0.0)
        scale_from_width = requested_scale

        max_scale_w = base_available_w / wm_w
        max_scale_h = base_available_h / wm_h
        max_scale = min(max_scale_w, max_scale_h)

        effective_scale = min(scale_from_width, max_scale)
        if effective_scale <= 0.0:
            return self._reassemble(src_rgb, src_alpha, src_extra, src_img_tensor)

        new_w = max(1, int(round(wm_w * effective_scale)))
        new_h = max(1, int(round(wm_h * effective_scale)))

        wm_color_resized, wm_alpha_resized = self._resize_rgba(wm_rgba, new_h, new_w)
        wm_color_resized = wm_color_resized.clamp(0.0, 1.0)
        wm_alpha_resized = wm_alpha_resized.clamp(0.0, 1.0)

        wm_layer_rgb = torch.zeros(
            (src_h, src_w, 3), device=src_rgb.device, dtype=src_rgb.dtype
        )
        wm_layer_alpha = torch.zeros(
            (src_h, src_w, 1), device=src_rgb.device, dtype=src_rgb.dtype
        )

        x_pos = self._compute_position(src_w, new_w, padding)
        y_pos = self._compute_position(src_h, new_h, padding)

        wm_layer_rgb[y_pos : y_pos + new_h, x_pos : x_pos + new_w, :] = wm_color_resized
        wm_layer_alpha[y_pos : y_pos + new_h, x_pos : x_pos + new_w, :] = wm_alpha_resized

        wm_layer_alpha = wm_layer_alpha.clamp(0.0, 1.0)

        result_rgb = src_rgb * (1.0 - wm_layer_alpha) + wm_layer_rgb * wm_layer_alpha

        if src_alpha is not None:
            result_alpha = src_alpha * (1.0 - wm_layer_alpha) + wm_layer_alpha
        else:
            result_alpha = None

        return self._reassemble(result_rgb, result_alpha, src_extra, src_img_tensor)

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
    def _split_channels(img_tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        channels = img_tensor.shape[-1]
        if channels == 1:
            rgb = img_tensor.repeat(1, 1, 3)
            alpha = None
            extra = None
        elif channels == 2:
            rgb = img_tensor[..., :1].repeat(1, 1, 3)
            alpha = img_tensor[..., 1:2]
            extra = None
        elif channels == 3:
            rgb = img_tensor[..., :3]
            alpha = None
            extra = None
        elif channels >= 4:
            rgb = img_tensor[..., :3]
            alpha = img_tensor[..., 3:4]
            extra = img_tensor[..., 4:] if channels > 4 else None
        else:
            raise ValueError("Unsupported number of channels in source image")
        return rgb, alpha, extra

    @staticmethod
    def _ensure_rgba(wm_tensor: torch.Tensor) -> torch.Tensor:
        channels = wm_tensor.shape[-1]
        if channels == 4:
            return wm_tensor
        if channels == 3:
            alpha = torch.ones_like(wm_tensor[..., :1])
            return torch.cat([wm_tensor, alpha], dim=-1)
        if channels == 1:
            repeated = wm_tensor.repeat(1, 1, 3)
            alpha = torch.ones_like(wm_tensor[..., :1])
            return torch.cat([repeated, alpha], dim=-1)
        raise ValueError("Unsupported number of channels in watermark image")

    @staticmethod
    def _resize_rgba(
        rgba_tensor: torch.Tensor, new_h: int, new_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        color = rgba_tensor[..., :3].permute(2, 0, 1).unsqueeze(0)
        alpha = rgba_tensor[..., 3:4].permute(2, 0, 1).unsqueeze(0)

        alpha = alpha.clamp(0.0, 1.0)
        premultiplied = color * alpha

        resized_alpha = F.interpolate(
            alpha, size=(new_h, new_w), mode="bicubic", align_corners=False
        )
        resized_color = F.interpolate(
            premultiplied, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        eps = 1e-6
        resized_alpha_clamped = resized_alpha.clamp(min=0.0, max=1.0)
        safe_alpha = resized_alpha_clamped.clamp_min(eps)
        unpremultiplied = resized_color / safe_alpha
        unpremultiplied = torch.where(
            resized_alpha_clamped > eps, unpremultiplied, torch.zeros_like(unpremultiplied)
        )

        resized_color = unpremultiplied.clamp(0.0, 1.0)
        resized_alpha = resized_alpha_clamped

        resized_color = resized_color.squeeze(0).permute(1, 2, 0)
        resized_alpha = resized_alpha.squeeze(0).permute(1, 2, 0)

        return resized_color, resized_alpha

    @staticmethod
    def _compute_position(base: int, layer: int, padding: int) -> int:
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
        return result.clamp(0.0, 1.0).to(
            dtype=template.dtype, device=template.device
        )


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}
