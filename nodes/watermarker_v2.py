import numpy as np
import torch
from PIL import Image


class OCS_WatermarkerV2:
    """Overlay a watermark onto the bottom-right corner of an image while
    preserving the transparency of the watermark."""

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
        src_pil = self._tensor_to_pil(src_img_tensor)
        target_mode = src_pil.mode
        src_rgba = src_pil.convert("RGBA")

        wm_rgba = self._tensor_to_pil(wm_tensor).convert("RGBA")

        if scale_percent <= 0.0 or wm_rgba.width == 0 or wm_rgba.height == 0:
            composite = src_rgba
        else:
            scale_ratio = max(scale_percent / 100.0, 0.0)
            target_w = max(1, int(round(src_rgba.width * scale_ratio)))
            target_h = max(1, int(round(src_rgba.height * scale_ratio)))

            width_ratio = target_w / wm_rgba.width
            height_ratio = target_h / wm_rgba.height
            resize_ratio = min(width_ratio, height_ratio)

            new_w = max(1, int(round(wm_rgba.width * resize_ratio)))
            new_h = max(1, int(round(wm_rgba.height * resize_ratio)))

            resized = self._resize_with_alpha(wm_rgba, (new_w, new_h))

            watermark_layer = Image.new("RGBA", src_rgba.size, (0, 0, 0, 0))

            x = max(0, src_rgba.width - new_w - padding)
            y = max(0, src_rgba.height - new_h - padding)

            mask = resized.split()[3]
            watermark_layer.paste(resized, (x, y), mask)

            composite = Image.alpha_composite(src_rgba, watermark_layer)

        final_img = composite.convert(target_mode)
        return self._pil_to_tensor(final_img, src_img_tensor.dtype)

    # ------------------------------------------------------------------
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
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return Image.fromarray(array)
        raise ValueError("Unsupported image tensor shape for conversion to PIL image")

    @staticmethod
    def _pil_to_tensor(image, dtype):
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        array = array / 255.0
        tensor = torch.from_numpy(array)
        return tensor.to(dtype)

    @staticmethod
    def _resize_with_alpha(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image.copy()

        rgba = np.array(image, dtype=np.float32)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]

        alpha_norm = alpha / 255.0
        premultiplied = rgb * alpha_norm[..., None]

        rgb_img = Image.fromarray(np.clip(premultiplied, 0, 255).astype(np.uint8), mode="RGB")
        alpha_img = Image.fromarray(alpha.astype(np.uint8), mode="L")

        rgb_resized = rgb_img.resize(size, Image.LANCZOS)
        alpha_resized = alpha_img.resize(size, Image.LANCZOS)

        rgb_resized_arr = np.asarray(rgb_resized, dtype=np.float32)
        alpha_resized_arr = np.asarray(alpha_resized, dtype=np.float32)

        alpha_norm_resized = alpha_resized_arr / 255.0
        safe_alpha = np.clip(alpha_norm_resized, 1e-6, 1.0)

        unpremultiplied = np.zeros((*size[::-1], 3), dtype=np.float32)
        mask = alpha_norm_resized > 1e-6
        unpremultiplied[mask] = rgb_resized_arr[mask] / safe_alpha[mask, None]

        result = np.dstack(
            [np.clip(unpremultiplied, 0, 255).astype(np.uint8), alpha_resized_arr.astype(np.uint8)]
        )

        return Image.fromarray(result, mode="RGBA")


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}
