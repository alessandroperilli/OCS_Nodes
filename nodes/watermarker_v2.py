import numpy as np
import torch
from PIL import Image, ImageEnhance


try:  # Pillow 9.1+
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLING_BOX = Image.Resampling.BOX
except AttributeError:  # Pillow < 9.1
    RESAMPLING_LANCZOS = Image.LANCZOS
    RESAMPLING_BOX = Image.BOX


class OCS_WatermarkerV2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                "watermark_mask": ("MASK",),
                "corner": (
                    (
                        "bottom-right",
                        "bottom-left",
                        "top-right",
                        "top-left",
                    ),
                ),
                "corner_padding": ("INT", {"default": 25, "min": 0, "max": 8192}),
                "percent_of_image": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("watermarked_image",)
    FUNCTION = "apply"
    CATEGORY = "OCS Nodes"

    # ------------------------------------------------------------------
    def apply(
        self,
        source_image,
        watermark_image,
        watermark_mask,
        corner,
        corner_padding,
        percent_of_image,
        opacity,
    ):

        corner_padding = self._extract_scalar(corner_padding, int)
        percent_of_image = self._extract_scalar(percent_of_image, float)
        opacity = float(self._extract_scalar(opacity, float))
        corner = str(self._extract_scalar(corner, str)).strip().lower()

        src_tensor = self._ensure_tensor(source_image)

        wm_tensor = self._ensure_tensor(watermark_image)
        mask_tensor = self._ensure_tensor(watermark_mask)

        if wm_tensor.shape[0] != mask_tensor.shape[0]:
            # Allow mask batches of size 1 to be broadcast; otherwise sizes must match.
            if mask_tensor.shape[0] != 1:
                raise ValueError("watermark_image and watermark_mask batch sizes must match.")

        results = []

        for idx, src_img in enumerate(src_tensor):
            wm_idx = idx % wm_tensor.shape[0]
            mask_idx = wm_idx if mask_tensor.shape[0] > 1 else 0

            wm_img = self._tensor_to_pil(wm_tensor[wm_idx]).convert("RGB")
            mask_img = self._tensor_to_pil(mask_tensor[mask_idx]).convert("L")

            if wm_img.size != mask_img.size:
                raise ValueError("watermark_image and watermark_mask must have the same resolution.")

            rgba_watermark = self._merge_mask(wm_img, mask_img)

            result = self._add_image_watermark(
                self._tensor_to_pil(src_img),
                rgba_watermark,
                opacity,
                percent_of_image,
                corner_padding,
                corner,
            )

            results.append(self._pil_to_tensor(result, src_img.dtype))

        stacked = torch.stack(results, dim=0)
        return (stacked.to(device=src_tensor.device),)

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_mask(rgb_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        rgba = Image.new("RGBA", rgb_image.size)
        rgb_data = rgb_image.getdata()
        mask_data = mask_image.getdata()

        rgba_data = []
        for rgb, alpha in zip(rgb_data, mask_data):
            rgba_data.append(rgb + (255 - alpha,))

        rgba.putdata(rgba_data)
        return rgba

    # ------------------------------------------------------------------
    def _add_image_watermark(
        self,
        original: Image.Image,
        watermark: Image.Image,
        opacity: float,
        percent_of_image: float,
        corner_padding: int,
        corner: str,
    ) -> Image.Image:
        src_mode = original.mode
        base = original.convert("RGBA")

        scale_ratio = max(percent_of_image / 100.0, 0.0)

        if scale_ratio <= 0.0:
            return original

        target_w = max(1, int(round(base.width * scale_ratio)))
        target_h = max(1, int(round(base.height * scale_ratio)))

        width_ratio = target_w / watermark.width if watermark.width else 0
        height_ratio = target_h / watermark.height if watermark.height else 0
        resize_ratio = min(width_ratio, height_ratio) if width_ratio and height_ratio else 0

        if resize_ratio <= 0:
            return original

        new_w = max(1, int(round(watermark.width * resize_ratio)))
        new_h = max(1, int(round(watermark.height * resize_ratio)))

        resized = self._resize_rgba_with_premultiply(watermark, (new_w, new_h))

        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")

        alpha = resized.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(max(0.0, min(1.0, opacity)))
        resized.putalpha(alpha)

        x, y = self._resolve_corner_position(
            base.size,
            resized.size,
            corner_padding,
            corner,
        )

        composite = base.copy()
        composite.paste(resized, (x, y), resized)

        return composite.convert(src_mode)

    # ------------------------------------------------------------------
    @staticmethod
    def _resize_with_quality(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Resize with high-quality settings, guarding for Pillow versions."""

        try:
            return image.resize(size, RESAMPLING_LANCZOS, reducing_gap=2.0)
        except TypeError:
            # reducing_gap added in Pillow 9.1; fall back when unavailable.
            return image.resize(size, RESAMPLING_LANCZOS)

    # ------------------------------------------------------------------
    def _resize_rgba_with_premultiply(
        self,
        image: Image.Image,
        size: tuple[int, int],
    ) -> Image.Image:
        """Resize an RGBA image using pre-multiplied alpha to avoid edge halos."""

        if image.size == size:
            return image.copy()

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Convert to numpy for accurate pre-multiplication in floating point.
        arr = np.asarray(image, dtype=np.float32)
        alpha = arr[..., 3:4]

        # Normalize alpha to [0, 1] and pre-multiply RGB channels.
        alpha_norm = alpha / 255.0
        premultiplied = arr.copy()
        premultiplied[..., :3] *= alpha_norm

        premultiplied_image = Image.fromarray(
            np.clip(np.round(premultiplied), 0, 255).astype(np.uint8),
            mode="RGBA",
        )

        resized = self._progressive_downscale(premultiplied_image, size)

        # Convert back to float for un-premultiplication.
        resized_arr = np.asarray(resized, dtype=np.float32)
        resized_alpha = resized_arr[..., 3:4]

        # Avoid divide-by-zero by forcing a minimum alpha where necessary.
        safe_alpha = np.where(resized_alpha > 0, resized_alpha, 1.0)
        unpremultiplied_rgb = resized_arr[..., :3] * (255.0 / safe_alpha)

        # Clamp RGB to valid bounds and zero-out fully transparent pixels.
        unpremultiplied_rgb = np.clip(unpremultiplied_rgb, 0.0, 255.0)
        unpremultiplied_rgb = np.where(resized_alpha > 0, unpremultiplied_rgb, 0.0)

        output = np.concatenate([unpremultiplied_rgb, resized_alpha], axis=-1)

        return Image.fromarray(
            np.clip(np.round(output), 0, 255).astype(np.uint8),
            mode="RGBA",
        )

    # ------------------------------------------------------------------
    def _progressive_downscale(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Downscale using progressive box filtering followed by LANCZOS.

        Using a multi-step approach with a box filter before the final
        high-quality resize preserves more detail when shrinking large
        watermarks.
        """

        target_w, target_h = target_size

        if image.size == target_size:
            return image.copy()

        if target_w >= image.width and target_h >= image.height:
            return self._resize_with_quality(image, target_size)

        current = image

        while True:
            next_w = max(target_w, current.width // 2)
            next_h = max(target_h, current.height // 2)

            if next_w == current.width and next_h == current.height:
                break

            # If halving would overshoot in either dimension, stop early.
            if next_w <= target_w and next_h <= target_h:
                break

            current = current.resize(
                (next_w, next_h),
                RESAMPLING_BOX,
            )

        if current.size != target_size:
            current = self._resize_with_quality(current, target_size)

        return current

    # ------------------------------------------------------------------
    def _resolve_corner_position(
        self,
        base_size: tuple[int, int],
        watermark_size: tuple[int, int],
        padding: int,
        corner: str,
    ) -> tuple[int, int]:
        base_w, base_h = base_size
        wm_w, wm_h = watermark_size

        pad = max(0, padding)

        normalized_corner = (corner or "bottom-right").lower()
        if normalized_corner not in {
            "top-left",
            "bottom-left",
            "top-right",
            "bottom-right",
        }:
            normalized_corner = "bottom-right"

        if normalized_corner == "top-left":
            return (pad, pad)
        if normalized_corner == "bottom-left":
            return (pad, max(0, base_h - wm_h - pad))
        if normalized_corner == "top-right":
            return (max(0, base_w - wm_w - pad), pad)

        return (max(0, base_w - wm_w - pad), max(0, base_h - wm_h - pad))

    @staticmethod
    def _ensure_tensor(img):
        if isinstance(img, list):
            img = img[0]
        if isinstance(img, torch.Tensor):
            tensor = img
        else:
            tensor = torch.as_tensor(img)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.unsqueeze(0)

        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor

    @staticmethod
    def _extract_scalar(value, caster):
        if isinstance(value, (list, tuple)):
            value = value[0]
        return caster(value)

    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        array = img_tensor.detach().cpu().clamp(0, 1).numpy()
        array = (array * 255.0).round().astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array.squeeze(-1)
        return Image.fromarray(array)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        array = array / 255.0
        tensor = torch.from_numpy(array)
        return tensor.to(dtype)


NODE_CLASS_MAPPINGS = {
    "OCS_WatermarkerV2": OCS_WatermarkerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_WatermarkerV2": "Watermarker v2",
}

