"""Pure released-variant source, query-mapping and wheel-constraint contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from geo_infer_test import dependency_audit, torch_variant

pytestmark = pytest.mark.unit


@pytest.fixture
def variant_workspace(tmp_path: Path) -> tuple[Path, dict]:
    """Keep ordinary and official-variant Torch as distinct universal nodes."""
    contract = torch_variant.official_variant_contract()
    index = {
        "name": contract["index_name"],
        "url": contract["source"]["registry"],
        "explicit": True,
    }
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="geo-infer-framework"\nversion="0.4.0"\n'
        "[tool.uv]\nconstraint-dependencies=["
        + json.dumps(torch_variant.CONSTRAINT)
        + "]\n"
        '[tool.uv.sources]\ntorch={index="pytorch-cu126",marker='
        + json.dumps(contract["marker"])
        + "}\n"
        '[[tool.uv.index]]\nname="pytorch-cu126"\nurl='
        + json.dumps(index["url"])
        + "\nexplicit=true\n"
    )
    (tmp_path / "uv.toml").write_text(
        'required-version="==0.13.1"\nconstraint-dependencies=['
        + json.dumps(torch_variant.CONSTRAINT)
        + ']\n[[index]]\nname="pytorch-cu126"\nurl='
        + json.dumps(index["url"])
        + "\nexplicit=true\n"
    )
    (tmp_path / ".python-version").write_text("3.12\n")
    member = tmp_path / "GEO-INFER-TEST"
    member.mkdir()
    (member / "pyproject.toml").write_text(
        '[project]\nname="geo-infer-test"\nversion="0.4.0"\n'
    )
    wheels = []
    for minor in ("311", "312", "315"):
        filename = f"torch-2.13.0+cu126-cp{minor}-cp{minor}-manylinux_2_28_x86_64.whl"
        row = contract["wheels"][filename]
        wheels.append({"url": row["url"], "hash": "sha256:" + row["sha256"]})
    package = {
        "name": "torch",
        "version": contract["version"],
        "source": contract["source"],
        "wheels": wheels,
    }
    prefix = (
        'version=1\n[[package]]\nname="geo-infer-framework"\nversion="0.4.0"\nsource={virtual="."}\n'
        '[[package]]\nname="geo-infer-test"\nversion="0.4.0"\nsource={editable="GEO-INFER-TEST"}\n'
        '[[package]]\nname="torch"\nversion="2.13.0"\nsource={registry="https://pypi.org/simple"}\n'
        '[[package]]\nname="torch"\nversion="2.13.0+cu126"\nsource={registry="https://download.pytorch.org/whl/cu126"}\n'
    )
    prefix += (
        "wheels=["
        + ",".join(
            "{url=" + json.dumps(row["url"]) + ",hash=" + json.dumps(row["hash"]) + "}"
            for row in wheels
        )
        + "]\n"
    )
    (tmp_path / "uv.lock").write_text(prefix)
    return tmp_path, package


def test_variant_keeps_full_identity_and_public_query(variant_workspace):
    root, package = variant_workspace
    record = torch_variant.official_variant_record(root, package)
    assert record["locked_identity"]["version"] == "2.13.0+cu126"
    assert record["advisory_identity"] == {"name": "torch", "version": "2.13.0"}
    pairs, local, hashes, nodes = dependency_audit.advisory_inventory(root)
    assert pairs == [("torch", "2.13.0")]
    assert len(nodes) == 2 and len(local) == 2
    assert {node["locked_identity"]["version"] for node in nodes} == {
        "2.13.0",
        "2.13.0+cu126",
    }
    assert "uv.lock" in hashes and any(
        path.endswith("torch_cu126_variant.json") for path in hashes
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("version", "2.13.0+unreviewed"),
        ("source", {"registry": "https://private.invalid/simple"}),
    ],
)
def test_no_generic_registry_or_local_version_mapping(variant_workspace, field, value):
    root, package = variant_workspace
    package = copy.deepcopy(package)
    package[field] = value
    with pytest.raises(ValueError):
        torch_variant.official_variant_record(root, package)


@pytest.mark.parametrize("mode", ["hash", "url", "duplicate", "missing312", "sdist"])
def test_variant_refuses_unbound_wheel_inventory(variant_workspace, mode):
    root, package = variant_workspace
    package = copy.deepcopy(package)
    if mode in ("hash", "url"):
        package["wheels"][0][mode] += "changed"
    elif mode == "duplicate":
        package["wheels"].append(package["wheels"][0])
    elif mode == "missing312":
        package["wheels"] = [
            row for row in package["wheels"] if "cp312-cp312" not in row["url"]
        ]
    else:
        package["sdist"] = {"url": "https://invalid/source.tar.gz"}
    with pytest.raises(ValueError):
        torch_variant.official_variant_record(root, package)


@pytest.mark.parametrize("minor", [11, 12, 15])
def test_constraint_selects_exact_locked_current_tag(variant_workspace, minor):
    root, _ = variant_workspace
    target = {
        "implementation": "cpython",
        "version": [3, minor],
        "sys_platform": "linux",
        "machine": "x86_64",
        "soabi": f"cpython-3{minor}-x86_64-linux-gnu",
    }
    result = torch_variant.locked_torch_constraint(root, target)
    assert (
        f"cp3{minor}-cp3{minor}-manylinux_2_28_x86_64.whl#sha256="
        in result["requirement"]
    )
    filename = f"torch-2.13.0+cu126-cp3{minor}-cp3{minor}-manylinux_2_28_x86_64.whl"
    released = torch_variant.official_variant_contract()["wheels"][filename]
    assert (
        result["requirement"]
        == "torch @ " + released["url"] + "#sha256=" + released["sha256"]
    )
    assert result["locked_identity"]["version"] == "2.13.0+cu126"
    assert len(result["lock_sha256"]) == 64
    target["soabi"] = "unreviewed"
    with pytest.raises(ValueError):
        torch_variant.locked_torch_constraint(root, target)


def test_other_platforms_never_receive_variant_constraint(tmp_path):
    for system, machine in (
        ("darwin", "arm64"),
        ("win32", "AMD64"),
        ("linux", "aarch64"),
    ):
        assert (
            torch_variant.locked_torch_constraint(
                tmp_path, {"sys_platform": system, "machine": machine}
            )
            is None
        )


def test_member_override_cannot_change_variant_ownership(variant_workspace):
    root, package = variant_workspace
    path = root / "GEO-INFER-TEST/pyproject.toml"
    path.write_text(path.read_text() + '[tool.uv.sources]\ntorch={index="other"}\n')
    with pytest.raises(ValueError):
        torch_variant.official_variant_record(root, package)


@pytest.mark.parametrize(
    "mode", ["marker", "project-constraint", "effective-constraint", "index", "tag"]
)
def test_source_and_tag_binding_fail_closed(variant_workspace, mode):
    root, package = variant_workspace
    if mode == "effective-constraint":
        path = root / "uv.toml"
        path.write_text(
            path.read_text().replace(torch_variant.CONSTRAINT, "torch==2.14.0")
        )
    elif mode == "tag":
        package = copy.deepcopy(package)
        package["wheels"][0]["url"] = package["wheels"][0]["url"].replace(
            "cp311-cp311", "cp311-cp311t"
        )
    else:
        path = root / "pyproject.toml"
        text = path.read_text()
        if mode == "marker":
            text = text.replace(
                "platform_machine == 'x86_64'", "platform_machine == 'aarch64'"
            )
        elif mode == "project-constraint":
            text = text.replace(torch_variant.CONSTRAINT, "torch==2.14.0")
        else:
            text = text.replace("explicit=true", "explicit=false")
        path.write_text(text)
    with pytest.raises(ValueError):
        torch_variant.official_variant_record(root, package)


@pytest.mark.parametrize(
    "version,soabi",
    [
        ([3, 13], "cpython-313-x86_64-linux-gnu"),
        ([3, 14], "cpython-314-x86_64-linux-gnu"),
        ([3, 16], "cpython-316-x86_64-linux-gnu"),
        ([3, 15], "cpython-315t-x86_64-linux-gnu"),
        ([3, 15], "cpython-315-x86_64-linux-musl"),
    ],
)
def test_tooling_constraint_refuses_unreviewed_version_or_abi(
    variant_workspace, version, soabi
):
    """Normal 3.15 tooling admission cannot widen other versions or ABI forms."""
    root, _ = variant_workspace
    target = {
        "implementation": "cpython",
        "version": version,
        "sys_platform": "linux",
        "machine": "x86_64",
        "soabi": soabi,
    }
    with pytest.raises(ValueError, match="Unreviewed Linux Torch installation ABI"):
        torch_variant.locked_torch_constraint(root, target)


def test_tooling_constraint_refuses_missing_315_lock_wheel(variant_workspace):
    """Required native rows cannot substitute for the selected tooling wheel."""
    root, _ = variant_workspace
    filename = "torch-2.13.0+cu126-cp315-cp315-manylinux_2_28_x86_64.whl"
    released = torch_variant.official_variant_contract()["wheels"][filename]
    entry = (
        "{url="
        + json.dumps(released["url"])
        + ",hash="
        + json.dumps("sha256:" + released["sha256"])
        + "}"
    )
    path = root / "uv.lock"
    text = path.read_text()
    assert text.count("," + entry) == 1
    path.write_text(text.replace("," + entry, ""))
    target = {
        "implementation": "cpython",
        "version": [3, 15],
        "sys_platform": "linux",
        "machine": "x86_64",
        "soabi": "cpython-315-x86_64-linux-gnu",
    }
    with pytest.raises(
        ValueError, match="Reviewed Torch wheel for target ABI is missing"
    ):
        torch_variant.locked_torch_constraint(root, target)
