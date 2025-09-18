"""Node that outputs the first non-empty value among two inputs."""

from ..helpers import any as ANY_TYPE


def _looks_like_context(value):
    return isinstance(value, dict) and "model" in value and "clip" in value


def _is_context_empty(ctx):
    return not ctx or all(val is None for val in ctx.values())


def _is_effectively_none(value):
    if value is None:
        return True
    if _looks_like_context(value):
        return _is_context_empty(value)
    return False


class OCS_FirstNotEmpty:
    """Outputs the first input that carries data, mirroring Any Switch with two sockets."""

    CATEGORY = "OCS Nodes"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "first": (ANY_TYPE,),
                "second": (ANY_TYPE,),
            },
        }

    def pick(self, first=None, second=None):
        for candidate in (first, second):
            if not _is_effectively_none(candidate):
                return (candidate,)
        return (None,)


NODE_CLASS_MAPPINGS = {
    "OCS_FirstNotEmpty": OCS_FirstNotEmpty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_FirstNotEmpty": "First Not Empty",
}
