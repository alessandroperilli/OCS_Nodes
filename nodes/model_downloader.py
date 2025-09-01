import os
import shutil
from typing import Optional

import requests

try:
    # ComfyUI provides this for pushing progress to the UI
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # Fallback for environments without ComfyUI server

try:
    import folder_paths  # type: ignore
    DEFAULT_MODELS_DIR: Optional[str] = getattr(folder_paths, "models_dir", None)
except Exception:
    folder_paths = None  # type: ignore
    DEFAULT_MODELS_DIR = None


class OCS_ModelDownloader:
    """Downloads a file from a URL to any user-specified folder.

    - Accepts absolute or relative paths and creates the folder if missing.
    - Supports optional bearer token (or env var via $VARNAME) for private URLs.
    - Reports download progress to ComfyUI if available.
    """

    # -------------- UI schema --------------
    @classmethod
    def INPUT_TYPES(cls):
        default_folder = DEFAULT_MODELS_DIR or os.getcwd()
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "https://huggingface.co/perilli/OCS_Models/resolve/main/VAE/Image/flux1_vae.safetensors",
                    },
                ),
                "folder": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": default_folder,
                        "tooltip": "Target directory (created if missing). Absolute or relative.",
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "flux1_vae.safetensors",
                    },
                ),
            },
            "optional": {
                "token": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "password": True,
                        "tooltip": "Optional bearer token. Use $VARNAME to pull from environment.",
                    },
                ),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FILE_PATH",)
    FUNCTION = "download"
    OUTPUT_NODE = True
    CATEGORY = "OCS Nodes"

    # -------------- core logic --------------
    def __init__(self):
        self.node_id = None

    def _send_progress(self, value: float, max_value: int = 100):
        if PromptServer is None or self.node_id is None:
            return
        try:
            PromptServer.instance.send_sync(
                "progress", {"node": self.node_id, "value": value, "max": max_value}
            )
        except Exception:
            pass

    def download(self, url: str, folder: str, filename: str, node_id: str, token: str = ""):
        self.node_id = node_id

        if not url or not filename:
            print(f"[OCS_ModelDownloader] Missing required values: url='{url}', filename='{filename}'")
            return ("",)

        # Expand env vars (~, $VAR, %VAR% on Windows) and normalize path
        folder_expanded = os.path.expanduser(os.path.expandvars(folder or ""))
        if not os.path.isabs(folder_expanded):
            folder_expanded = os.path.abspath(folder_expanded)

        # Ensure target directory exists
        try:
            os.makedirs(folder_expanded, exist_ok=True)
        except Exception as e:
            print(f"[OCS_ModelDownloader] Cannot create target folder '{folder_expanded}': {e}")
            return ("",)

        save_path = os.path.join(folder_expanded, filename)
        if os.path.exists(save_path):
            print(f"[OCS_ModelDownloader] File already exists: {save_path}")
            return (save_path,)

        # Expand token from environment if it starts with '$'
        if token.startswith("$"):
            env_value = os.getenv(token[1:])
            token = env_value if env_value is not None else token

        headers = {"Authorization": f"Bearer {token}"} if token else None

        print(
            f"[OCS_ModelDownloader] Downloading {url} to {save_path}"
            + (" with Authorization header" if headers else "")
        )

        try:
            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                temp_path = save_path + ".tmp"

                downloaded = 0
                last_report = 0.0

                with open(temp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        size = file.write(chunk)
                        downloaded += size

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100.0
                            if progress - last_report >= 0.2:
                                print(
                                    f"[OCS_ModelDownloader] Downloading {filename}... {progress:.1f}%"
                                )
                                self._send_progress(progress, 100)
                                last_report = progress

                # Finalize
                shutil.move(temp_path, save_path)
                if total_size > 0:
                    self._send_progress(100.0, 100)
                print(f"[OCS_ModelDownloader] Complete! Saved to {save_path}")
                return (save_path,)

        except Exception as e:
            # Clean up partial download
            try:
                temp_path = save_path + ".tmp"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            print(f"[OCS_ModelDownloader] Error: {e}")
            return ("",)


NODE_CLASS_MAPPINGS = {
    "OCS_ModelDownloader": OCS_ModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_ModelDownloader": "Model Downloader",
}