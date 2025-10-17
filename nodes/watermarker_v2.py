import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

import folder_paths


class OCS_WatermarkerV2:

    @classmethod
    def INPUT_TYPES(cls):
        font_dir = cls._font_directory()
        font_files: Iterable[str] = []

        if font_dir is not None and font_dir.exists():
            font_files = sorted(
                f for f in os.listdir(font_dir) if (font_dir / f).is_file()
            )

        return {
            "required": {
                "source_image": ("IMAGE",),
                "use_image_watermark": ("BOOLEAN", {"default": True}),
                "scale_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "padding": ("INT", {"default": 25, "min": 0, "max": 8192}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "watermark_image": ("IMAGE",),
                "watermark_mask": ("MASK",),
                "text": ("STRING", {"default": "enter text", "multiline": False}),
                "text_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "font_name": ((list(font_files),),),
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
        use_image_watermark,
        scale_percent,
        padding,
        opacity,
        watermark_image=None,
        watermark_mask=None,
        text="enter text",
        text_color="#FFFFFF",
        font_name=None,
    ):

        padding = self._extract_scalar(padding, int)
        scale_percent = self._extract_scalar(scale_percent, float)
        opacity = float(self._extract_scalar(opacity, float))
        use_image_watermark = bool(self._extract_scalar(use_image_watermark, bool))

        src_tensor = self._ensure_tensor(source_image)

        if use_image_watermark:
            if watermark_image is None or watermark_mask is None:
                raise ValueError("Both watermark_image and watermark_mask are required.")

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
                    scale_percent,
                    padding,
                )

                results.append(self._pil_to_tensor(result, src_img.dtype))

        else:
            font = self._load_font(font_name, scale_percent, src_tensor.shape)
            rgb = self._hex_to_rgb(text_color)

            results = []
            for src_img in src_tensor:
                src_pil = self._tensor_to_pil(src_img)

                watermarked = self._add_text_watermark(
                    src_pil,
                    str(text),
                    scale_percent,
                    opacity,
                    rgb,
                    font,
                    padding,
                )

                results.append(self._pil_to_tensor(watermarked, src_img.dtype))

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
    @staticmethod
    def _add_image_watermark(
        original: Image.Image,
        watermark: Image.Image,
        opacity: float,
        scale_percent: float,
        padding: int,
    ) -> Image.Image:
        src_mode = original.mode
        base = original.convert("RGBA")

        scale_ratio = max(scale_percent / 100.0, 0.0)

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

        resized = watermark.resize((new_w, new_h), Image.LANCZOS)

        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")

        alpha = resized.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(max(0.0, min(1.0, opacity)))
        resized.putalpha(alpha)

        x = max(0, base.width - new_w - padding)
        y = max(0, base.height - new_h - padding)

        composite = base.copy()
        composite.paste(resized, (x, y), resized)

        return composite.convert(src_mode)

    # ------------------------------------------------------------------
    def _add_text_watermark(
        self,
        original: Image.Image,
        text: str,
        scale_percent: float,
        opacity: float,
        color: tuple[int, int, int],
        font: ImageFont.FreeTypeFont,
        padding: int,
    ) -> Image.Image:
        src_mode = original.mode
        base = original.convert("RGBA")

        txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        font = font or self._derive_font(scale_percent, base.size)
        alpha_value = int(round(max(0.0, min(1.0, opacity)) * 255))

        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        x = max(0, base.width - text_width - padding)
        y = max(0, base.height - text_height - padding)

        draw.text((x, y), text, font=font, fill=(*color, alpha_value))

        composite = Image.alpha_composite(base, txt_layer)
        return composite.convert(src_mode)

    # ------------------------------------------------------------------
    @classmethod
    def _font_directory(cls) -> Optional[Path]:
        base = folder_paths.get_output_directory()
        custom = Path(base).parent / "custom_nodes" / "ComfyUI-MingNodes" / "fonts"
        return custom if custom.exists() else None

    def _load_font(self, font_name, scale_percent, src_shape):
        if not font_name:
            return None

        font_dir = self._font_directory()
        if font_dir is None:
            return None

        font_path = font_dir / font_name
        if not font_path.exists():
            return None

        size = self._font_size_from_scale(scale_percent, src_shape)
        size = max(1, size)

        try:
            return ImageFont.truetype(font_path.as_posix(), size)
        except OSError:
            return None

    def _derive_font(self, scale_percent: float, image_size):
        fallback_size = max(1, int(round(min(image_size) * max(scale_percent / 100.0, 0.01))))
        try:
            return ImageFont.truetype("DejaVuSans.ttf", fallback_size)
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def _font_size_from_scale(scale_percent: float, src_shape) -> int:
        if isinstance(src_shape, torch.Size):
            height = src_shape[-3]
            width = src_shape[-2]
        else:
            _, height, width, _ = src_shape
        base_dim = min(width, height)
        return int(round(base_dim * max(scale_percent / 100.0, 0.01)))

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) != 6:
            return (255, 255, 255)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

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

