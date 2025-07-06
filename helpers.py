import torch

#Credit to pythongosssss for the AnyType class
class AnyType(str):
    def __ne__(self, __value: object) -> bool:  
        return False

any = AnyType("*")


# User-friendly labels for node inputs
def _get_kw(label: str, kwargs: dict, args: tuple, pos: int):
    """Return the widget value whether Comfy passed it as kw‑ or positional arg."""
    if label in kwargs:
        return kwargs[label]
    if len(args) > pos:
        return args[pos]
    raise TypeError(f"Missing required argument: {label}")