import torch
from ..helpers import any, _get_kw

class OCS_CloudImageSize:

    @classmethod
    def INPUT_TYPES(cls):
        image_presets = [
            "------ OpenAI GPT-image-1 ------",
            "1536x1024 (3:2 | 1.6MP)",
            "1024x1536 (2:3 | 1.6MP)",
            "1024x1024 (1:1 | 1MP)",
        ]

        return {
            "required": {
                "Aspect Ratio": (image_presets,),
                # "Custom Width":  ("INT", {"default": 64, "min": 64, "max": 8192}),
                # "Custom Height": ("INT", {"default": 64, "min": 64, "max": 8192}),
                "Batch Size":  ("INT", {"default": 1, "min": 1,  "max": 64}),
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT SPECIFICATION
    # ──────────────────────────────────────────────────────────────────────────
    RETURN_TYPES = (
        any, # aspect_ratio (dimension string for combo inputs)
        "INT",    # image_width
        "INT",    # image_height
        "LATENT", # image_latent
        "INT",    # batch_size
    )

    RETURN_NAMES = (
        "aspect_ratio",
        "image_width",
        "image_height",
        "image_latent",
        "batch_size",
    )

    FUNCTION = "configure_sizes"
    CATEGORY = "OCS Nodes"

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL MAPPINGS
    # ──────────────────────────────────────────────────────────────────────────
    _IMAGE_MAP = {
        "1536x1024 (3:2 | 1.6MP)": (1536, 1024),
        "1024x1536 (2:3 | 1.6MP)": (1024, 1536),
        "1024x1024 (1:1 | 1MP)": (1024, 1024),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOGIC
    # ──────────────────────────────────────────────────────────────────────────
    def configure_sizes(self, *args, **kwargs):  # noqa: D401
        aspect_ratio = _get_kw("Aspect Ratio", kwargs, args, 0)
        # image_width  = _get_kw("Custom Width",  kwargs, args, 1)
        # image_height = _get_kw("Custom Height", kwargs, args, 2)
        batch_size   = _get_kw("Batch Size",    kwargs, args, 1) # was 3

        # Override custom image dims if preset chosen
        if aspect_ratio in self._IMAGE_MAP:
            image_width, image_height = self._IMAGE_MAP[aspect_ratio]

        # Latent tensors (SD‑type models require /8 dims)
        image_latent = torch.zeros([batch_size, 4, image_height // 8, image_width // 8])

        # Pure dimension string for combo output (first token before space)
        aspect_str = f"{image_width}x{image_height}" # if aspect_ratio == "custom" else aspect_ratio.split(" ")[0]

        return (
            aspect_str,
            image_width,
            image_height,
            {"samples": image_latent},
            batch_size,
        )


NODE_CLASS_MAPPINGS = {
    "OCS_CloudImageSize": OCS_CloudImageSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_CloudImageSize": "Image Size (Cloud Models)",
}
