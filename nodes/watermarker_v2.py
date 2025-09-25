import numpy as np
import torch
from PIL import Image
from typing import Optional

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:  # pragma: no cover - Pillow < 9 compatibility
    _LANCZOS = Image.LANCZOS


class OCS_WatermarkerV2:
    """Overlay a watermark onto the bottom-right corner of an image while
    preserving the transparency of the watermark by compositing directly in
    torch with proper alpha math."""

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

        src_tensor = self._ensure_tensor(source_image)
        wm_tensor = self._ensure_tensor(watermark)

        batch_size = src_tensor.shape[0]
        wm_count = wm_tensor.shape[0]

        results = []

        for idx in range(batch_size):
            src_img = src_tensor[idx]
            wm_img = wm_tensor[idx % wm_count]
            overlay = self._overlay_watermark(src_img, wm_img, scale_percent, padding)
            results.append(overlay)

        stacked = torch.stack(results, dim=0).to(device=src_tensor.device)
        return (stacked,)

    # ------------------------------------------------------------------
    def _overlay_watermark(self, src_img_tensor, wm_tensor, scale_percent, padding):
        if scale_percent <= 0.0:
            return src_img_tensor

        src = src_img_tensor.to(dtype=torch.float32)
        wm = wm_tensor.to(dtype=torch.float32)

        src_h, src_w = src.shape[0], src.shape[1]
        wm_h, wm_w = wm.shape[0], wm.shape[1]

        if wm_h == 0 or wm_w == 0:
            return src_img_tensor

        src_rgb, src_alpha, src_extra = self._split_channels(src)
        wm_rgba = self._ensure_rgba(wm)

        base_image = self._tensor_to_pil_rgba(src_rgb, src_alpha)
        watermark_image = self._tensor_to_pil_rgba(wm_rgba[..., :3], wm_rgba[..., 3:4])

        scale_ratio = max(scale_percent / 100.0, 0.0)
        target_w = max(1, int(round(src_w * scale_ratio)))
        target_h = max(1, int(round(src_h * scale_ratio)))

        width_ratio = target_w / wm_w
        height_ratio = target_h / wm_h
        resize_ratio = min(width_ratio, height_ratio)

        new_w = max(1, int(round(wm_w * resize_ratio)))
        new_h = max(1, int(round(wm_h * resize_ratio)))

        if watermark_image.size != (new_w, new_h):
            watermark_image = watermark_image.resize((new_w, new_h), _LANCZOS)

        x = max(0, src_w - new_w - padding)
        y = max(0, src_h - new_h - padding)

        watermark_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        watermark_layer.paste(watermark_image, (x, y), watermark_image)

        composited = Image.alpha_composite(base_image, watermark_layer)

        composited_tensor = self._pil_rgba_to_tensor(composited)

        result_rgb = composited_tensor[..., :3]
        result_alpha = composited_tensor[..., 3:4] if src_alpha is not None else None

        result = result_rgb
        if result_alpha is not None:
            result = torch.cat([result, result_alpha], dim=-1)
        if src_extra is not None:
            result = torch.cat([result, src_extra.detach().cpu()], dim=-1)

        return result.clamp(0.0, 1.0).to(dtype=src_img_tensor.dtype, device=src_img_tensor.device)

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
    def _split_channels(img_tensor: torch.Tensor):
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
    def _ensure_rgba(wm_tensor: torch.Tensor):
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
    def _tensor_to_pil_rgba(
        rgb_tensor: torch.Tensor, alpha_tensor: Optional[torch.Tensor]
    ) -> Image.Image:
        rgb_np = (
            rgb_tensor.clamp(0.0, 1.0)
            .detach()
            .cpu()
            .numpy()
        )
        rgb_bytes = (rgb_np * 255.0 + 0.5).astype(np.uint8)

        if alpha_tensor is not None:
            alpha_np = (
                alpha_tensor.clamp(0.0, 1.0)
                .detach()
                .cpu()
                .numpy()
            )
            alpha_bytes = (alpha_np * 255.0 + 0.5).astype(np.uint8)
        else:
            alpha_bytes = np.full(rgb_bytes.shape[:2] + (1,), 255, dtype=np.uint8)

        rgba = np.concatenate([rgb_bytes, alpha_bytes], axis=-1)
        return Image.fromarray(rgba, mode="RGBA")

    @staticmethod
    def _pil_rgba_to_tensor(image: Image.Image) -> torch.Tensor:
        rgba = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgba)
        return tensor


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}
