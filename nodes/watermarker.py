import numpy as np
import torch
from PIL import Image


class OCS_Watermarker:

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

    # ──────────────────────────────────────────────────────────────────────────
    def apply_watermark(self,
                        source_image,
                        watermark,
                        scale_percent,
                        padding):

        padding = self._extract_scalar(padding, int)
        scale_percent = self._extract_scalar(scale_percent, float)

        src_tensor = self._ensure_tensor(source_image)
        wm_tensor = self._ensure_tensor(watermark)

        batch_size = src_tensor.shape[0]
        wm_count = wm_tensor.shape[0]

        result = []

        for idx in range(batch_size):
            src_img = src_tensor[idx]
            wm_img = wm_tensor[idx % wm_count]

            watermarked = self._overlay_watermark(src_img, wm_img, scale_percent, padding)
            result.append(watermarked)

        stacked = torch.stack(result, dim=0)
        stacked = stacked.to(device=src_tensor.device)
        return (stacked,)

    # ──────────────────────────────────────────────────────────────────────────
    def _overlay_watermark(self, src_img_tensor, wm_tensor, scale_percent, padding):
        src_pil = self._tensor_to_pil(src_img_tensor)
        src_rgba = src_pil.convert("RGBA")

        wm_pil = self._tensor_to_pil(wm_tensor).convert("RGBA")

        if scale_percent <= 0.0 or wm_pil.width == 0 or wm_pil.height == 0:
            composite = src_rgba
        else:
            scale_ratio = max(scale_percent / 100.0, 0.0)

            target_w = max(1, int(round(src_rgba.width * scale_ratio)))
            target_h = max(1, int(round(src_rgba.height * scale_ratio)))

            width_ratio = target_w / wm_pil.width
            height_ratio = target_h / wm_pil.height
            resize_ratio = min(width_ratio, height_ratio)

            # Guarantee at least one pixel for extremely small percentages.
            new_w = max(1, int(round(wm_pil.width * resize_ratio)))
            new_h = max(1, int(round(wm_pil.height * resize_ratio)))

            resized = wm_pil.resize((new_w, new_h), Image.LANCZOS)

            composite = src_rgba.copy()

            x = max(0, src_rgba.width - new_w - padding)
            y = max(0, src_rgba.height - new_h - padding)

            alpha = resized.getchannel("A")
            composite.paste(resized, (x, y), alpha)

        target_mode = src_pil.mode
        final_img = composite.convert(target_mode)

        return self._pil_to_tensor(final_img, src_img_tensor.dtype)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_tensor(img):
        if isinstance(img, list):
            img = img[0]
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.dtype != torch.float32:
            img = img.float()
        return img

    @staticmethod
    def _extract_scalar(value, caster):
        if isinstance(value, list):
            value = value[0]
        return caster(value)

    @staticmethod
    def _tensor_to_pil(img_tensor):
        array = img_tensor.detach().cpu().clamp(0, 1).numpy()
        array = (array * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _pil_to_tensor(image, dtype):
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        array = array / 255.0
        tensor = torch.from_numpy(array)
        return tensor.to(dtype)


NODE_CLASS_MAPPINGS = {
    "OCS_Watermarker": OCS_Watermarker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_Watermarker": "Watermarker",
}
