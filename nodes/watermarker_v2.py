import torch
import torch.nn.functional as F


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

        scale_ratio = max(scale_percent / 100.0, 0.0)
        target_w = max(1, int(round(src_w * scale_ratio)))
        target_h = max(1, int(round(src_h * scale_ratio)))

        width_ratio = target_w / wm_w
        height_ratio = target_h / wm_h
        resize_ratio = min(width_ratio, height_ratio)

        new_w = max(1, int(round(wm_w * resize_ratio)))
        new_h = max(1, int(round(wm_h * resize_ratio)))

        resized = self._resize_rgba(wm_rgba, new_w, new_h)

        x = max(0, src_w - new_w - padding)
        y = max(0, src_h - new_h - padding)

        paste_w = min(new_w, src_w - x)
        paste_h = min(new_h, src_h - y)

        if paste_w <= 0 or paste_h <= 0:
            return src_img_tensor

        blended_rgb = src_rgb.clone()
        roi_rgb = blended_rgb[y : y + paste_h, x : x + paste_w, :]

        wm_color = resized[:paste_h, :paste_w, :3]
        wm_alpha = resized[:paste_h, :paste_w, 3:4]

        roi_rgb.mul_(1.0 - wm_alpha).add_(wm_color * wm_alpha)
        blended_rgb[y : y + paste_h, x : x + paste_w, :] = roi_rgb

        if src_alpha is not None:
            blended_alpha = src_alpha.clone()
            roi_alpha = blended_alpha[y : y + paste_h, x : x + paste_w, :]
            roi_alpha.mul_(1.0 - wm_alpha).add_(wm_alpha)
            blended_alpha[y : y + paste_h, x : x + paste_w, :] = roi_alpha
        else:
            blended_alpha = None

        result = blended_rgb
        if blended_alpha is not None:
            result = torch.cat([result, blended_alpha], dim=-1)
        if src_extra is not None:
            result = torch.cat([result, src_extra], dim=-1)

        return result.clamp(0.0, 1.0).to(dtype=src_img_tensor.dtype)

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
    def _resize_rgba(wm: torch.Tensor, new_w: int, new_h: int):
        height, width = wm.shape[0], wm.shape[1]
        if width == new_w and height == new_h:
            return wm

        rgba = wm.clamp(0.0, 1.0)
        color = rgba[..., :3] * rgba[..., 3:4]
        alpha = rgba[..., 3:4]

        color_4d = color.permute(2, 0, 1).unsqueeze(0)
        alpha_4d = alpha.permute(2, 0, 1).unsqueeze(0)

        interpolate_kwargs = {
            "size": (new_h, new_w),
            "mode": "bicubic",
            "align_corners": False,
        }

        try:
            resized_color = F.interpolate(
                color_4d, antialias=True, **interpolate_kwargs
            )
            resized_alpha = F.interpolate(
                alpha_4d, antialias=True, **interpolate_kwargs
            )
        except TypeError:
            # Older torch builds do not support the antialias flag. Fall back to the
            # default behaviour instead of raising so the node remains compatible.
            resized_color = F.interpolate(color_4d, **interpolate_kwargs)
            resized_alpha = F.interpolate(alpha_4d, **interpolate_kwargs)

        resized_alpha = resized_alpha.clamp(0.0, 1.0)
        safe_alpha = resized_alpha.clamp_min(1e-6)

        resized_color = torch.where(
            resized_alpha > 1e-6,
            resized_color / safe_alpha,
            torch.zeros_like(resized_color),
        )

        resized_color = resized_color.clamp(0.0, 1.0)

        color_hw = resized_color.squeeze(0).permute(1, 2, 0)
        alpha_hw = resized_alpha.squeeze(0).permute(1, 2, 0)

        return torch.cat([color_hw, alpha_hw], dim=-1)


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}
