#!/usr/bin/env python3

"""
Tests for the geo_infer_agent command-line interface: create-config,
load_config, and load_agent_class for every advertised agent type.
"""

import argparse
import os
import tempfile

import yaml

from geo_infer_agent import cli
from geo_infer_agent.core.agent_registry import AgentRegistry


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"type": None, "output": None, "force": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCreateConfig:
    """create-config writes real, loadable configuration templates."""

    def _write(self, agent_type):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "cfg.yaml")
            cli.create_config_command(_args(type=agent_type, output=output))
            with open(output) as f:
                return yaml.safe_load(f)

    def test_writes_template_for_every_advertised_type(self) -> None:
        # Every type the CLI can run must have a real template.
        assert set(cli.CONFIG_TEMPLATES) == set(cli.AGENT_MODULES)
        for agent_type in cli.CONFIG_TEMPLATES:
            config = self._write(agent_type)
            assert config["agent_type"] == agent_type

    def test_default_template_is_loadable_by_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "cfg.yaml")
            cli.create_config_command(_args(type="default", output=output))
            loaded = cli.load_config(output)
            assert loaded["agent_type"] == "default"

    def test_existing_file_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "cfg.yaml")
            cli.create_config_command(_args(type="default", output=output))
            with open(output) as f:
                original = f.read()
            try:
                cli.create_config_command(_args(type="bdi", output=output))
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("must refuse to overwrite without --force")
            with open(output) as f:
                assert f.read() == original  # untouched

            cli.create_config_command(_args(type="bdi", output=output, force=True))
            with open(output) as f:
                assert yaml.safe_load(f)["agent_type"] == "bdi"


class TestLoadAgentClass:
    """Every advertised agent type resolves to a real, instantiable class."""

    def test_all_types_resolve(self) -> None:
        for agent_type in cli.AGENT_MODULES:
            agent_class = cli.load_agent_class(agent_type)
            assert isinstance(agent_class, type)

    def test_unknown_type_exits(self) -> None:
        try:
            cli.load_agent_class("no-such-type")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("unknown type must exit(1)")

    def test_registry_knows_the_same_types(self) -> None:
        AgentRegistry._instance = None
        registry = AgentRegistry()
        assert set(registry.agent_types) == set(cli.AGENT_MODULES)

    def test_type_to_path_mapping_has_a_single_definition(self) -> None:
        # The CLI must not hand-copy the registry's mapping: both must be
        # backed by the same module-level AGENT_TYPES constant.
        import geo_infer_agent.core.agent_registry as ar

        assert cli.AGENT_MODULES is ar.AGENT_TYPES

    def test_cli_registry_and_list_descriptions_agree(self) -> None:
        AgentRegistry._instance = None
        registry = AgentRegistry()
        assert (
            set(cli.AGENT_MODULES)
            == set(registry.agent_types)
            == set(cli.AGENT_DESCRIPTIONS)
        )
