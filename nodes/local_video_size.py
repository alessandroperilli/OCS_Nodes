import torch
from ..helpers import any, _get_kw

class OCS_LocalVideoSize:

    @classmethod
    def INPUT_TYPES(cls):
        video_presets = [
            "custom",
            "1360x768 [CogVideoX 1.5]",
            "1280x720 [WanVideo 2.1, Hunyuan Video]",
            "960x544 [Hunyuan Video]",
            "854x480 [WanVideo 2.1]",
            "720x480 [CogVideoX 1.5]",
        ]

        return {
            "required": {
                "Aspect Ratio": (video_presets,),
                "Custom Width":  ("INT", {"default": 64, "min": 64, "max": 8192}),
                "Custom Height": ("INT", {"default": 64,  "min": 64, "max": 8192}),
                "Batch Size":  ("INT", {"default": 1, "min": 1,  "max": 64}),
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT SPECIFICATION
    # ──────────────────────────────────────────────────────────────────────────
    RETURN_TYPES = (
        any, # aspect_ratio (dimension string for combo inputs)
        "INT",    # video_width
        "INT",    # video_height
        "LATENT", # video_latent
        "INT",    # batch_size
    )

    RETURN_NAMES = (
        "aspect_ratio",
        "video_width",
        "video_height",
        "video_latent",
        "batch_size",
    )

    FUNCTION = "configure_sizes"
    CATEGORY = "OCS Nodes"

    _VIDEO_MAP = {
        "1360x768 [CogVideoX 1.5]": (1360, 768),
        "1280x720 [WanVideo 2.1, Hunyuan Video]":  (1280, 720),
        "960x544 [Hunyuan Video]":  (960, 544),
        "854x480 [WanVideo 2.1]":   (854, 480),
        "720x480 [CogVideoX 1.5]":  (720, 480),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOGIC
    # ──────────────────────────────────────────────────────────────────────────
    def configure_sizes(self, *args, **kwargs):  # noqa: D401
        aspect_ratio = _get_kw("Aspect Ratio", kwargs, args, 0)
        video_width  = _get_kw("Custom Width",  kwargs, args, 1)
        video_height = _get_kw("Custom Height", kwargs, args, 2)
        batch_size   = _get_kw("Batch Size",    kwargs, args, 3)

        # Override custom video dims if preset chosen
        if aspect_ratio in self._VIDEO_MAP:
            video_width, video_height = self._VIDEO_MAP[aspect_ratio]

        # Latent tensors (SD‑type models require /8 dims)
        video_latent = torch.zeros([batch_size, 4, video_height // 8, video_width // 8])

        # Pure dimension string for combo output (first token before space)
        aspect_str = f"{video_width}x{video_height}" if aspect_ratio == "custom" else aspect_ratio.split(" ")[0]

        return (
            aspect_str,
            video_width,
            video_height,
            {"samples": video_latent},
            batch_size,
        )


NODE_CLASS_MAPPINGS = {
    "OCS_LocalVideoSize": OCS_LocalVideoSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_LocalVideoSize": "Video Size (Local Models)",
}
