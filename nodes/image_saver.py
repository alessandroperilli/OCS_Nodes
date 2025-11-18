from ..helpers import any, _get_kw
import torch

import json
import os, sys
import numpy as np
from pathlib import Path
from datetime import datetime
import folder_paths
from PIL import Image, PngImagePlugin
import piexif


class OCS_ImageSaver:

    INPUT_IS_LIST = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    # -------------------------- UI --------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "filename": (
                    "STRING",
                    {"default": "%seed_%date_%time_final_OCS", "multiline": False},
                ),
                "path": ("STRING", {"default": "%date/", "multiline": False}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "image_format": (["png", "jpg", "jpeg", "webp"],),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "jpg_webp_quality": (
                    "INT",
                    {"default": 100, "min": 1, "max": 100},
                ),
                "date_format": ("STRING", {"default": "%Y-%m-%d", "multiline": False}),
                "time_format": ("STRING", {"default": "%H%M%S", "multiline": False}),
                "EXIF_UserComment": ("STRING", {"default": "", "multiline": True}),
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embeds the complete workflow data into the image metadata. Only works with PNG and WebP formats."
                    }),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("FILENAME", "FILE_PATH")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "OCS Nodes"

    # ---------------------- helpers ------------------------
    @staticmethod
    def _strftime(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    # ------------------- main routine ----------------------
    def save_images(
        self,
        images,
        filename: str = "%seed_%date_%time_final_meta",
        path: str = "%date/",
        seed: int = 0,
        image_format: str = "png",
        lossless_webp: bool = True,
        jpg_webp_quality: int = 100,
        date_format: str = "%Y-%m-%d",
        time_format: str = "%H%M%S",
        embed_workflow: bool = True,
        EXIF_UserComment: str = "",
        extra_pnginfo=None,
    ):

        flat_images = self._flatten_images(images)
        if not flat_images:
            raise ValueError("No images provided to OCS_ImageSaver")

        # unpack widget scalars when Comfy wraps them in single-element lists
        seed = self._unwrap_scalar(seed)
        filename = self._unwrap_scalar(filename)
        path = self._unwrap_scalar(path)
        image_format = self._unwrap_scalar(image_format)
        lossless_webp = self._unwrap_scalar(lossless_webp)
        jpg_webp_quality = self._unwrap_scalar(jpg_webp_quality)
        date_format = self._unwrap_scalar(date_format)
        time_format = self._unwrap_scalar(time_format)
        embed_workflow = self._unwrap_scalar(embed_workflow)
        EXIF_UserComment = self._unwrap_scalar(EXIF_UserComment)

        (
            full_output_folder,
            filename_alt,
            counter_alt,
            subfolder_alt,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            self.prefix_append,
            self.output_dir,
            flat_images[0].shape[1],
            flat_images[0].shape[0],
        )

        output_folder = Path(full_output_folder)
        counter_base = counter_alt

        base_vars = {
            "%date": self._strftime(date_format),
            "%time": self._strftime(time_format),
            "%seed": seed,
            "%image_format": image_format,
        }

        saved_filenames, saved_paths, ui_images = [], [], []

        for (batch_number, image) in enumerate(flat_images):
            var_map = base_vars.copy()
            var_map["%counter"] = f"{counter_base + batch_number:05}"

            rel_folder = self._replace_tokens(path, var_map)
            rel_filename = self._replace_tokens(filename, var_map)

            final_folder = output_folder / rel_folder
            final_folder.mkdir(parents=True, exist_ok=True)

            full_path = final_folder / f"{rel_filename}.{image_format}"

            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            self.process_image(
                img,
                full_path,
                image_format,
                lossless_webp,
                jpg_webp_quality,
                embed_workflow,
                EXIF_UserComment,
                seed,
                extra_pnginfo=extra_pnginfo,
            )

            saved_filenames.append(full_path.name)
            saved_paths.append(str(full_path))
            ui_images.append(
                {
                    "filename": full_path.name,
                    "subfolder": str(rel_folder),
                    "type": self.type,
                }
            )

            print(f"[OCS_ImageSaver] Saved: {full_path}")

        return {
            "ui": {"images": ui_images},
            "result": (self._single_or_list(saved_filenames), self._single_or_list(saved_paths)),
        }

    # -------------------- internals ------------------------
    @staticmethod
    def _single_or_list(lst):
        return lst[0] if len(lst) == 1 else lst

    @staticmethod
    def _unwrap_scalar(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    @staticmethod
    def _replace_tokens(template: str, mapping: dict) -> str:
        for k, v in mapping.items():
            template = template.replace(k, str(v))
        return template.strip("/")

    @staticmethod
    def _flatten_images(images):
        """Normalise IMAGE, batch, or (possibly nested) lists into a flat list."""

        if isinstance(images, torch.Tensor):
            # single image or explicit batch tensor
            return list(images) if images.ndim == 4 else [images]

        flat = []

        def _walk(item):
            if isinstance(item, torch.Tensor):
                flat.extend(list(item) if item.ndim == 4 else [item])
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    _walk(sub)
            elif item is None:
                # ignore empty placeholders often produced by optional inputs
                return
            else:
                raise TypeError(
                    "OCS_ImageSaver expects IMAGE tensors, batches, or lists thereof"
                )

        _walk(images)
        return flat

    @staticmethod
    def process_image(
        img: Image.Image,
        path: Path,
        image_format: str,
        lossless_webp: bool,
        quality: int,
        embed_workflow: bool,
        EXIF_UserComment: str,
        seed: int,
        extra_pnginfo=None,
    ):

        try:
            exif_bytes = None

            if EXIF_UserComment:
                try:
                    # Store user comment with UTF-16BE encoding per EXIF spec
                    exif_payload = {
                        "Exif": {
                            piexif.ExifIFD.UserComment: b"UNICODE\0" + EXIF_UserComment.encode("utf-16be")
                        }
                    }
                    exif_bytes = piexif.dump(exif_payload)
                except Exception as e:
                    print(f"Error adding EXIF data: {e}")

            save_kwargs = {}
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes

            if image_format == "png":
                pnginfo = PngImagePlugin.PngInfo()

                if embed_workflow and extra_pnginfo is not None:
                    workflow_json = json.dumps(extra_pnginfo["workflow"])
                    pnginfo.add_text("workflow", workflow_json)

                img.save(path, pnginfo=pnginfo, **save_kwargs)

            elif image_format == "webp":
                img.save(path, "WEBP", lossless=lossless_webp, quality=quality, **save_kwargs)

            elif image_format in {"jpg", "jpeg"}:
                img.convert("RGB").save(path, quality=quality, **save_kwargs)

        except Exception as e:
            print(f"Error saving image: {e}")


NODE_CLASS_MAPPINGS = {
    "OCS_ImageSaver": OCS_ImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_ImageSaver": "Image Saver",
}
