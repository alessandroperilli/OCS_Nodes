"""Workflow analysis utilities for OCS Nodes."""

from __future__ import annotations

from collections import defaultdict
import sys
from typing import Dict, Iterable

Workflow = Dict[str, object]
NodeDict = Dict[str, object]


class OCS_NodesContribution:
    """Summarise how many nodes in the active workflow come from each suite."""

    NAME = "Nodes Contribution"
    CATEGORY = "OCS Nodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "main"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Counts the nodes used in the current workflow grouped by their source suite,"
        " including suites that are installed but not referenced."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "workflow": "WORKFLOW",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _iter_node_class_mappings() -> Dict[str, type]:
        """Collect *all* NODE_CLASS_MAPPINGS exposed by loaded modules."""

        combined: Dict[str, type] = {}
        for module in list(sys.modules.values()):
            mapping = getattr(module, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    combined.setdefault(key, value)
        return combined

    @staticmethod
    def _suite_from_python_module(rel_module: str | None) -> str:
        if not rel_module:
            return "Unknown"

        parts = rel_module.split(".")
        if not parts:
            return "Unknown"

        head = parts[0]
        if head == "custom_nodes" and len(parts) > 1:
            head = parts[1]
        elif head in {"nodes", "comfy"}:
            return "Comfy Core"

        if head == "comfy_extras":
            return "Comfy Extras"
        if head == "comfy_api_nodes":
            return "Comfy API Nodes"

        return head

    def _installed_suites(self) -> Dict[str, set[str]]:
        class_map = self._iter_node_class_mappings()
        suites: Dict[str, set[str]] = defaultdict(set)
        for node_id, cls in class_map.items():
            rel_module = getattr(cls, "RELATIVE_PYTHON_MODULE", None)
            suite = self._suite_from_python_module(rel_module)
            suites[suite].add(node_id)
        return suites

    @staticmethod
    def _workflow_nodes(workflow: Workflow | None) -> Iterable[NodeDict]:
        if isinstance(workflow, dict):
            nodes = workflow.get("nodes")
            if isinstance(nodes, list):
                return nodes
        return []

    # ------------------------------------------------------------------ main API
    @staticmethod
    def _set_widget_text(
        extra_pnginfo: Dict[str, object] | None,
        unique_id: str | int | None,
        value: str,
    ) -> None:
        if not extra_pnginfo or unique_id is None:
            return

        workflow_info = extra_pnginfo.get("workflow")
        if not isinstance(workflow_info, dict):
            return

        nodes = workflow_info.get("nodes")
        if not isinstance(nodes, list):
            return

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id")) == str(unique_id):
                node["widgets_values"] = [value]
                break

    def _ensure_text_widget(
        self,
        extra_pnginfo: Dict[str, object] | None,
        unique_id: str | int | None,
    ) -> None:
        """Guarantee a text widget exists so the node shows an empty panel on load."""

        if not extra_pnginfo or unique_id is None:
            return

        workflow_info = extra_pnginfo.setdefault("workflow", {})
        if not isinstance(workflow_info, dict):
            return

        nodes = workflow_info.setdefault("nodes", [])
        if not isinstance(nodes, list):
            return

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id")) == str(unique_id):
                current = node.setdefault("widgets_values", [])
                if isinstance(current, list):
                    if not current:
                        current.append("")
                else:
                    node["widgets_values"] = [""]
                return

    def main(
        self,
        workflow: Workflow | None = None,
        unique_id: str | int | None = None,
        extra_pnginfo: Dict[str, object] | None = None,
    ):
        self._ensure_text_widget(extra_pnginfo, unique_id)

        suites = self._installed_suites()
        if not suites:
            message = "No node suites detected."
            self._set_widget_text(extra_pnginfo, unique_id, message)
            return {"ui": {"text": (message,)}, "result": (message,)}

        class_to_suite: Dict[str, str] = {
            node_id: suite for suite, members in suites.items() for node_id in members
        }

        counts = {suite: 0 for suite in suites}
        unknown_count = 0

        for node in self._workflow_nodes(workflow):
            class_type = node.get("class_type") if isinstance(node, dict) else None
            if not class_type and isinstance(node, dict):
                class_type = node.get("type")

            if isinstance(class_type, str) and class_type in class_to_suite:
                counts[class_to_suite[class_type]] += 1
            elif class_type:
                unknown_count += 1

        if unknown_count:
            counts.setdefault("Unregistered", 0)
            counts["Unregistered"] += unknown_count

        sorted_lines = sorted(
            counts.items(), key=lambda item: (-item[1], item[0].lower())
        )
        breakdown_lines = [f"{suite} - {count}" for suite, count in sorted_lines]
        details_lines = ["", "Loaded node classes:"]
        for suite in sorted(suites):
            members = sorted(suites[suite])
            details_lines.append(
                f"{suite} ({len(members)} class{'es' if len(members) != 1 else ''})"
            )
            for node_id in members:
                details_lines.append(f"  • {node_id}")

        ui_text = "\n".join(breakdown_lines + details_lines)

        self._set_widget_text(extra_pnginfo, unique_id, ui_text)
        return {"ui": {"text": (ui_text,)}, "result": (ui_text,)}


NODE_CLASS_MAPPINGS = {
    "OCS_NodesContribution": OCS_NodesContribution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCS_NodesContribution": "Nodes Contribution",
}
