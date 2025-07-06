import torch
from ..helpers import any, _get_kw

class OCS_LocalImageSize:

    @classmethod
    def INPUT_TYPES(cls):
        image_presets = [
            "custom",
            "--------- FLUX.1, SD 3.5 ---------",
            "1152x1728 (2:3 | 2MP)",
            "1216x1664 (3:4 | 2MP)",
            "1728x1152 (3:2 | 2MP)",
            "1664x1216 (4:3 | 2MP)",
            "1920x1088 (16:9 | 2MP)",
            "2176x960 (21:9 | 2MP)",
            "1408x1408 (1:1 | 2MP)",
            "----- FLUX.1, SD 3.5, SDXL -----",
            "896x1152 (3:4 | 1MP)",
            "832x1216 (5:8 | 1MP)",
            "1152x896 (4:3 | 1MP)",
            "1216x832 (3:2 | 1MP)",
            "1344x768 (16:9 | 1MP)",
            "1536x640 (21:9 | 1MP)",
            "1024x1024 (1:1 | 1MP)",
            "------------- SD 1.5 ---------------",
            "512x768 (2:3 | 0.4MP)",
            "512x682 (3:4 | 0.3MP)",
            "768x512 (3:2 | 0.4MP)",
            "682x512 (4:3 | 0.3MP)",
            "910x512 (16:9 | 0.5MP)",
            "952x512 (1.85:1 | 0.5MP)",
            "512x512 (1:1 | 0.3MP)",
        ]

        return {
            "required": {
                "Aspect Ratio": (image_presets,),
                "Custom Width":  ("INT", {"default": 64, "min": 64, "max": 8192}),
                "Custom Height": ("INT", {"default": 64, "min": 64, "max": 8192}),
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

    _IMAGE_MAP = {
        #--------- FLUX.1, SD 3.5 ---------
        "1152x1728 (2:3 | 2MP)": (1152, 1728),
        "1216x1664 (3:4 | 2MP)": (1216, 1664),
        "1728x1152 (3:2 | 2MP)": (1728, 1152),
        "1664x1216 (4:3 | 2MP)": (1664, 1216),
        "1920x1088 (16:9 | 2MP)": (1920, 1088),
        "2176x960 (21:9 | 2MP)": (2176, 960),
        "1408x1408 (1:1 | 2MP)": (1408, 1408),
        #----- FLUX.1, SD 3.5, SDXL -----
        "896x1152 (3:4 | 1MP)": (896, 1152),
        "832x1216 (5:8 | 1MP)": (832, 1216),
        "1152x896 (4:3 | 1MP)": (1152, 896),
        "1216x832 (3:2 | 1MP)": (1216, 832),
        "1344x768 (16:9 | 1MP)": (1344, 768),
        "1536x640 (21:9 | 1MP)": (1536, 640),
        "1024x1024 (1:1 | 1MP)": (1024, 1024),
        #------------- SD 1.5 ---------------
        "512x768 (2:3 | 0.4MP)": (512, 768),
        "512x682 (3:4 | 0.3MP)": (512, 682),
        "768x512 (3:2 | 0.4MP)": (768, 512),
        "682x512 (4:3 | 0.3MP)": (682, 512),
        "910x512 (16:9 | 0.5MP)": (910, 512),
        "952x512 (1.85:1 | 0.5MP)": (952, 512),
        "512x512 (1:1 | 0.3MP)": (512, 512),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOGIC
    # ──────────────────────────────────────────────────────────────────────────
    def configure_sizes(self, *args, **kwargs):  # noqa: D401
        aspect_ratio = _get_kw("Aspect Ratio", kwargs, args, 0)
        image_width  = _get_kw("Custom Width",  kwargs, args, 1)
        image_height = _get_kw("Custom Height", kwargs, args, 2)
        batch_size   = _get_kw("Batch Size",    kwargs, args, 3)

        # Override custom image dims if preset chosen
        if aspect_ratio in self._IMAGE_MAP:
            image_width, image_height = self._IMAGE_MAP[aspect_ratio]

        # Latent tensors (SD‑type models require /8 dims)
        image_latent = torch.zeros([batch_size, 4, image_height // 8, image_width // 8])

        # Pure dimension string for combo output (first token before space)
        aspect_str = f"{image_width}x{image_height}" if aspect_ratio == "custom" else aspect_ratio.split(" ")[0]

        return (
            aspect_str,
            image_width,
            image_height,
            {"samples": image_latent},
            batch_size,
        )
        

NODE_CLASS_MAPPINGS = {
    "OCS_LocalImageSize": OCS_LocalImageSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_LocalImageSize": "Image Size (Local Models)",
}
