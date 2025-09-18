import torch


class OCS_ImageGrid4x4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_r1c1": ("IMAGE",),
            "image_r1c2": ("IMAGE",),
            "image_r1c3": ("IMAGE",),
            "image_r1c4": ("IMAGE",),
            "image_r2c1": ("IMAGE",),
            "image_r2c2": ("IMAGE",),
            "image_r2c3": ("IMAGE",),
            "image_r2c4": ("IMAGE",),
            "image_r3c1": ("IMAGE",),
            "image_r3c2": ("IMAGE",),
            "image_r3c3": ("IMAGE",),
            "image_r3c4": ("IMAGE",),
            "image_r4c1": ("IMAGE",),
            "image_r4c2": ("IMAGE",),
            "image_r4c3": ("IMAGE",),
            "image_r4c4": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "OCS Nodes"
    DESCRIPTION = """
Concatenates the 16 input images into a 4x4 grid.
"""

    def compositegrid(
        self,
        image_r1c1,
        image_r1c2,
        image_r1c3,
        image_r1c4,
        image_r2c1,
        image_r2c2,
        image_r2c3,
        image_r2c4,
        image_r3c1,
        image_r3c2,
        image_r3c3,
        image_r3c4,
        image_r4c1,
        image_r4c2,
        image_r4c3,
        image_r4c4,
    ):
        top_row = torch.cat((image_r1c1, image_r1c2, image_r1c3, image_r1c4), dim=2)
        second_row = torch.cat((image_r2c1, image_r2c2, image_r2c3, image_r2c4), dim=2)
        third_row = torch.cat((image_r3c1, image_r3c2, image_r3c3, image_r3c4), dim=2)
        bottom_row = torch.cat((image_r4c1, image_r4c2, image_r4c3, image_r4c4), dim=2)
        grid = torch.cat((top_row, second_row, third_row, bottom_row), dim=1)
        return (grid,)


NODE_CLASS_MAPPINGS = {
    "OCS_ImageGrid4x4": OCS_ImageGrid4x4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_ImageGrid4x4": "Image Grid 4x4",
}
