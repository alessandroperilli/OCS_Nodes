"""OCS Nodes package – auto‑loads every *.py in the *nodes* sub‑package
and aggregates their NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS.
"""

from importlib import import_module, reload
from pkgutil import iter_modules
from pathlib import Path
import sys

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

_pkg_dir = Path(__file__).parent / "nodes"

def _merge(child):
    NODE_CLASS_MAPPINGS.update(getattr(child, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(child, "NODE_DISPLAY_NAME_MAPPINGS", {}))

def _load_all():
    for info in iter_modules([_pkg_dir.as_posix()]):
        if info.name.startswith("_"):
            continue
        child = import_module(f"{__name__}.nodes.{info.name}")
        _merge(child)

_load_all()

# -------- optional hot‑reload hook --------
def refresh():
    """Reload all node modules. Hook this up to ComfyUI's Ctrl+R."""
    for name in list(sys.modules):
        if name.startswith(f"{__name__}.nodes."):
            reload(sys.modules[name])
    NODE_CLASS_MAPPINGS.clear()
    NODE_DISPLAY_NAME_MAPPINGS.clear()
    _load_all()