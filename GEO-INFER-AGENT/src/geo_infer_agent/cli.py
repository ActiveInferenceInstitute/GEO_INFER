#!/usr/bin/env python3

"""
Command-line interface for GEO-INFER-AGENT

This module provides a command-line interface for interacting with
the GEO-INFER-AGENT autonomous agent functionality.
"""

import os
import sys
import argparse
import logging
import yaml
import asyncio
import json
import importlib
from typing import Any, cast

from geo_infer_agent.core.agent_registry import AGENT_DESCRIPTIONS, AGENT_TYPES

# Configure logger
logger = logging.getLogger("geo_infer_agent.cli")


# Embedded configuration templates for ``create-config``.  Keys mirror the
# config values each agent class actually reads (see the corresponding
# module's __init__/config handling).  Kept in sync with load_agent_class().
CONFIG_TEMPLATES: dict[str, dict[str, Any]] = {
    "default": {
        "agent_type": "default",
        "memory_capacity": 1000,
        "decision_frequency": 5,
        "max_runtime": 86400,
    },
    "data_collector": {
        "agent_type": "data_collector",
        "storage_path": "data",
        "collection_interval": 300,
        "max_retries": 3,
        "timeout": 30,
        "region": "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        "data_sources": [],
    },
    "bdi": {
        "agent_type": "bdi",
        "memory_capacity": 1000,
        "deliberation_interval": 5,
        "commitment_strategy": "single_minded",
        "initial_beliefs": {},
        "initial_desires": [],
        "plans": [],
    },
    "active_inference": {
        "agent_type": "active_inference",
        "memory_capacity": 1000,
        "planning_horizon": 3,
        "state_dimensions": 10,
        "observation_dimensions": 10,
        "control_dimensions": 5,
    },
    "reinforcement_learning": {
        "agent_type": "reinforcement_learning",
        "memory_capacity": 1000,
        "state_size": 100,
        "action_size": 5,
        "buffer_capacity": 10000,
    },
    "rule_based": {
        "agent_type": "rule_based",
        "memory_capacity": 1000,
        "rules": [],
        "initial_facts": {},
    },
    "hybrid": {
        "agent_type": "hybrid",
        "memory_capacity": 1000,
    },
}


# Single definition lives in core.agent_registry (AGENT_TYPES); the CLI
# derives class paths and descriptions from it.  AGENT_MODULES is kept as
# an alias for existing importers.
AGENT_MODULES = AGENT_TYPES


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the CLI.

    Args:
        verbose: Whether to use verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config = json.load(f)
            else:
                logger.error(f"Unsupported file format: {config_path}")
                sys.exit(1)

        logger.info(f"Loaded configuration from {config_path}")
        return cast(dict[str, Any], config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)


def load_agent_class(agent_type: str) -> type:
    """
    Dynamically load agent class based on type.

    Args:
        agent_type: Type of agent to load

    Returns:
        Agent class
    """
    if agent_type not in AGENT_MODULES:
        logger.error(f"Unknown agent type: {agent_type}")
        logger.info(f"Available agent types: {', '.join(AGENT_MODULES)}")
        sys.exit(1)

    try:
        # Split module path and class name
        module_path, class_name = AGENT_MODULES[agent_type].rsplit(".", 1)

        # Import module
        module = importlib.import_module(module_path)

        # Get class
        agent_class = getattr(module, class_name)

        return cast(type, agent_class)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load agent class for type '{agent_type}': {str(e)}")
        logger.error("Make sure the required agent modules are installed")
        sys.exit(1)


async def run_agent(args: argparse.Namespace) -> None:
    """
    Run an agent instance.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    config_path = args.config
    if not config_path:
        logger.error("No configuration file specified")
        sys.exit(1)

    config = load_config(config_path)

    # Determine agent type
    agent_type = args.type or config.get("agent_type", "default")

    # Load agent class
    agent_class = load_agent_class(agent_type)

    try:
        # Create agent instance
        agent = agent_class(agent_id=args.id, config=config)

        # If state file provided, load state
        if args.state:
            if os.path.exists(args.state):
                loader = agent_class.load_state
                agent = loader(args.state, config=config)
                logger.info(f"Loaded agent state from {args.state}")
            else:
                logger.warning(f"State file not found: {args.state}")

        # Run agent
        logger.info(f"Starting agent {agent.agent_id} of type {agent_type}")
        await agent.run()

        # Save state if requested
        if args.save_state:
            state_path = agent.save_state(args.save_state)
            logger.info(f"Saved agent state to {state_path}")

    except KeyboardInterrupt:
        logger.info("Agent execution interrupted")
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        sys.exit(1)


def list_agents_command(args: argparse.Namespace) -> None:
    """
    List available agent types.

    Args:
        args: Command-line arguments
    """
    # Descriptions come from the registry's single-source constants, so
    # this listing can never drift from load_agent_class().
    agents = {
        agent_type: AGENT_DESCRIPTIONS.get(agent_type, "") for agent_type in AGENT_TYPES
    }

    print("Available Agent Types:\n")
    for agent_type, description in agents.items():
        print(f"  {agent_type:15s} - {description}")

    print("\nUse with: geo-infer-agent run --type <agent_type>")


def create_config_command(args: argparse.Namespace) -> None:
    """
    Create a configuration file template.

    Args:
        args: Command-line arguments
    """
    # Define the output path
    output_path = args.output or "agent_config.yaml"

    # Check if file already exists
    if os.path.exists(output_path) and not args.force:
        logger.error(f"File already exists: {output_path}. Use --force to overwrite.")
        sys.exit(1)

    # Determine which agent type to create config for
    agent_type = args.type or "default"
    if agent_type not in CONFIG_TEMPLATES:
        logger.error(
            "Unknown agent type: %s. Valid types: %s",
            agent_type,
            ", ".join(sorted(CONFIG_TEMPLATES)),
        )
        sys.exit(1)

    # Templates are embedded in code so they ship with the installed package
    # (packaging YAML data files alongside the wheel proved fragile).
    template = dict(CONFIG_TEMPLATES[agent_type])

    try:
        with open(output_path, "w") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created configuration template at {output_path}")

    except Exception as e:
        logger.error(f"Failed to create configuration file: {str(e)}")
        sys.exit(1)


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="GEO-INFER-AGENT - Autonomous agent framework for geospatial applications",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument(
        "--config", "-c", required=True, help="Path to configuration file"
    )
    run_parser.add_argument("--type", "-t", help="Agent type to run")
    run_parser.add_argument("--id", help="Agent ID (generated if not provided)")
    run_parser.add_argument("--state", "-s", help="Path to state file to load")
    run_parser.add_argument("--save-state", help="Path to save state after execution")

    # List command
    subparsers.add_parser("list", help="List available agent types")

    # Create-config command
    config_parser = subparsers.add_parser(
        "create-config", help="Create a configuration template"
    )
    config_parser.add_argument("--type", "-t", help="Agent type to create config for")
    config_parser.add_argument("--output", "-o", help="Output file path")
    config_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing file"
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "run":
        asyncio.run(run_agent(args))
    elif args.command == "list":
        list_agents_command(args)
    elif args.command == "create-config":
        create_config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
