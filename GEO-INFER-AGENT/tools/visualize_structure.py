#!/usr/bin/env python3

"""
GEO-INFER-AGENT Structure Visualizer

This script generates a visualization of the GEO-INFER-AGENT module structure
as a Markdown file with Mermaid diagrams.
"""

import os
import re
from datetime import datetime
from typing import Any

# Configuration
OUTPUT_FILE = "GEO-INFER-AGENT/docs/module_structure.md"
ROOT_DIR = "GEO-INFER-AGENT/src/geo_infer_agent"
EXCLUDED_DIRS = {".git", "__pycache__", "*.egg-info", ".pytest_cache"}
EXCLUDED_FILES = {"*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.egg", "*.swp"}


def should_exclude(path: str) -> bool:
    """Check if a path should be excluded."""
    name = os.path.basename(path)

    if os.path.isdir(path):
        return any(
            re.match(pattern.replace("*", ".*"), name) for pattern in EXCLUDED_DIRS
        )
    else:
        return any(
            re.match(pattern.replace("*", ".*"), name) for pattern in EXCLUDED_FILES
        )


def collect_structure(root_dir: str) -> dict[str, Any]:
    """Collect the directory structure."""
    result = {"name": os.path.basename(root_dir), "type": "dir", "children": []}

    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)

        if should_exclude(path):
            continue

        if os.path.isdir(path):
            result["children"].append(collect_structure(path))
        else:
            result["children"].append({"name": item, "type": "file"})

    return result


def generate_tree_markdown(structure: dict[str, Any], indent: int = 0) -> list[str]:
    """Generate a Markdown tree representation."""
    lines = []
    prefix = "  " * indent

    if structure["type"] == "dir":
        lines.append(f"{prefix}- 📁 **{structure['name']}/**")

        for child in structure["children"]:
            lines.extend(generate_tree_markdown(child, indent + 1))
    else:
        lines.append(f"{prefix}- 📄 {structure['name']}")

    return lines


def generate_mermaid_flowchart(structure: dict[str, Any]) -> list[str]:
    """Generate a Mermaid flowchart representation."""
    lines = ["```mermaid", "flowchart TD"]
    node_ids = {}

    def process_node(node: dict[str, Any], parent_id: str | None = None) -> None:
        node_name = node["name"]
        node_id = f"{parent_id}_{node_name}" if parent_id else node_name
        node_id = node_id.replace(".", "_").replace("-", "_").replace(" ", "_")

        if node["type"] == "dir":
            lines.append(f"    {node_id}[📁 {node_name}]")

            for child in node["children"]:
                child_id = process_node(child, node_id)
                lines.append(f"    {node_id} --> {child_id}")
        else:
            lines.append(f"    {node_id}[📄 {node_name}]")

        node_ids[node_id] = node_name
        return node_id

    process_node(structure)
    lines.append("```")

    return lines


def generate_class_diagram(root_dir: str) -> list[str]:
    """Generate a Mermaid class diagram for the models."""
    lines = ["```mermaid", "classDiagram"]

    # Define model classes
    models_dir = os.path.join(root_dir, "models")
    if not os.path.isdir(models_dir):
        return []

    models = [
        ("bdi", ["BDIAgent", "BDIState", "Belief", "Desire", "Plan"]),
        (
            "active_inference",
            ["ActiveInferenceAgent", "ActiveInferenceState", "GenerativeModel"],
        ),
        ("rl", ["RLAgent", "RLState", "QTable", "ReplayBuffer", "Experience"]),
        ("rule_based", ["RuleBasedAgent", "RuleBasedState", "Rule", "RuleSet"]),
        ("hybrid", ["HybridAgent", "HybridState", "SubAgentWrapper"]),
    ]

    # Add base classes
    lines.append("    class BaseAgent {")
    lines.append("        +id: str")
    lines.append("        +config: Dict")
    lines.append("        +initialize() async")
    lines.append("        +perceive() async")
    lines.append("        +decide() async")
    lines.append("        +act() async")
    lines.append("        +shutdown() async")
    lines.append("    }")

    lines.append("    class AgentState {")
    lines.append("        +agent_id: str")
    lines.append("        +to_dict()")
    lines.append("        +from_dict()")
    lines.append("    }")

    # Add model-specific classes
    for module, classes in models:
        for cls in classes:
            lines.append(f"    class {cls}")

    # Add relationships
    for module, classes in models:
        for cls in classes:
            if cls.endswith("Agent"):
                lines.append(f"    BaseAgent <|-- {cls}")
            elif cls.endswith("State"):
                lines.append(f"    AgentState <|-- {cls}")

        # Add module-specific relationships
        if module == "bdi":
            lines.append("    BDIAgent *-- BDIState")
            lines.append("    BDIState *-- Belief")
            lines.append("    BDIState *-- Desire")
            lines.append("    BDIState *-- Plan")
        elif module == "active_inference":
            lines.append("    ActiveInferenceAgent *-- ActiveInferenceState")
            lines.append("    ActiveInferenceState *-- GenerativeModel")
        elif module == "rl":
            lines.append("    RLAgent *-- RLState")
            lines.append("    RLState *-- QTable")
            lines.append("    RLState *-- ReplayBuffer")
            lines.append("    ReplayBuffer *-- Experience")
        elif module == "rule_based":
            lines.append("    RuleBasedAgent *-- RuleBasedState")
            lines.append("    RuleBasedState *-- RuleSet")
            lines.append("    RuleSet *-- Rule")
        elif module == "hybrid":
            lines.append("    HybridAgent *-- HybridState")
            lines.append("    HybridState *-- SubAgentWrapper")
            lines.append("    SubAgentWrapper *-- BaseAgent")

    lines.append("```")

    return lines


def main():
    """Main function."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Collect structure
    structure = collect_structure(ROOT_DIR)

    # Generate Markdown content
    content = [
        "# GEO-INFER-AGENT Module Structure",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Directory Structure",
        "",
    ]

    # Add tree representation
    content.extend(generate_tree_markdown(structure))
    content.append("")

    # Add flowchart
    content.append("## Module Diagram")
    content.append("")
    content.extend(generate_mermaid_flowchart(structure))
    content.append("")

    # Add class diagram
    content.append("## Class Diagram")
    content.append("")
    content.extend(generate_class_diagram(ROOT_DIR))

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(content))

    print(f"Structure visualization written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
