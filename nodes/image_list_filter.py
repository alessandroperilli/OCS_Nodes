import torch
from ..helpers import any, _get_kw

class OCS_ImageListFilter:
    """
    Passes through or removes images from an IMAGE list according
    to a minimum width / height. “0” means “no limit” for that dimension.

    • width_min: images with width  ≤ width_min  are dropped
    • height_min: images with height ≤ height_min are dropped

    If *all* images are dropped and a ``fallback_image`` is supplied, the node
    outputs a single‑element IMAGE list containing that fallback image.
    The fallback tensor is normalised (uint8 → float32 0‑1) and guaranteed to
    have an explicit batch‑dimension so that downstream nodes and PIL previews
    can handle it without raising a *Cannot handle this data type* error.
    """

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    OUTPUT_IS_LIST = (True, False)
    RETURN_NAMES = ("images", "removed_indices")
    FUNCTION = "filter"
    CATEGORY = "OCS Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width_min": ("INT", {"default": 0, "min": 0}),
                "height_min": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "fallback_image": ("IMAGE",),
            },
        }

    def _ensure_tensor_4d_float(self, img):
        """Make sure *img* is float32 with shape (1, H, W, C)."""
        if isinstance(img, list):  # Unwrap widget lists
            img = img[0]
        if img.ndim == 3:          # (H, W, C) → add batch dim
            img = img.unsqueeze(0)
        if img.dtype == torch.uint8:  # uint8 0‑255 → float32 0‑1
            img = img.to(torch.float32).div(255.0)
        return img

    def filter(self,
               images,
               width_min,
               height_min,
               fallback_image=None):

        # unwrap scalar widget lists (Comfy wraps INT widgets in 1‑elem lists)
        if isinstance(width_min, list):
            width_min = width_min[0]
        if isinstance(height_min, list):
            height_min = height_min[0]

        # Convert optional fallback right away so it’s ready if needed
        if fallback_image is not None:
            fallback_image = self._ensure_tensor_4d_float(fallback_image)

        # user entry N means “keep ≥ N+1” (to match original semantics)
        width_thr = width_min + 1 if width_min else 0
        height_thr = height_min + 1 if height_min else 0

        kept, removed = [], []

        for idx, img in enumerate(images):
            _, H, W, _ = img.shape  # B, H, W, C

            if (width_thr and W < width_thr) or (height_thr and H < height_thr):
                removed.append(idx)
            else:
                kept.append(img)

        # If nothing survived, fall back to a single image when provided
        if not kept and fallback_image is not None:
            kept.append(fallback_image)

        return kept, ", ".join(map(str, removed))


NODE_CLASS_MAPPINGS = {
    "OCS_ImageListFilter": OCS_ImageListFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_ImageListFilter": "Image List Filter",
}
