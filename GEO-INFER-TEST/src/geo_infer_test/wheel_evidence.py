"""Validate retained installed-wheel evidence without importing GEO runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .torch_variant import locked_torch_constraint, provenance_hashes


def installation_source(root: Path, target: dict, constraints: str) -> dict:
    """Bind an actual child ABI and exported pins to the single source owner."""
    if (
        not isinstance(target, dict)
        or set(target)
        != {"implementation", "version", "sys_platform", "machine", "soabi"}
        or any(
            not isinstance(target.get(k), str) or not target[k]
            for k in ("implementation", "sys_platform", "machine")
        )
        or not isinstance(target.get("version"), list)
        or len(target["version"]) != 2
        or any(type(n) is not int or n < 0 for n in target["version"])
        or (target.get("soabi") is not None and not isinstance(target["soabi"], str))
        or not isinstance(constraints, str)
        or not constraints
        or len(constraints.encode("utf-8")) > 4 * 1024**2
        or "\0" in constraints
    ):
        raise ValueError("Invalid wheel installation target or exported constraints")
    binding = locked_torch_constraint(root, target)
    text = binding["requirement"] + "\n" if binding is not None else ""
    lock_sha256 = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if binding is not None and binding["lock_sha256"] != lock_sha256:
        raise ValueError("Torch binding belongs to another lock")
    return {
        "schema": "geo-infer-wheel-installation-source/1",
        "target": target,
        "lock_sha256": lock_sha256,
        "exported_constraints": {
            "text": constraints,
            "sha256": hashlib.sha256(constraints.encode()).hexdigest(),
        },
        "variant_constraints": {
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
        "torch_binding": binding,
        "owner_sha256": {
            Path(path).name: digest for path, digest in provenance_hashes().items()
        },
    }


def installation_source_sha256(value: dict) -> str:
    """Use one deterministic digest for producer, child output and reader."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_installation_source(value: dict, *, root: Path, lock_sha256: str) -> str:
    """Refuse missing, altered or unauthenticated installed-source evidence."""
    if not isinstance(value, dict) or not isinstance(
        value.get("exported_constraints"), dict
    ):
        raise ValueError("Wheel installation-source role is missing or malformed")
    expected = installation_source(
        root, value.get("target"), value["exported_constraints"].get("text")
    )
    if value != expected or value["lock_sha256"] != lock_sha256:
        raise ValueError(
            "Wheel installation-source evidence differs from the frozen source contract"
        )
    return installation_source_sha256(value)


def validate_wheel_receipt(
    path: Path,
    *,
    modules: list[str],
    profiles: list[tuple[str, str, str]],
    revision: str | None = None,
    require_clean: bool = False,
    require_imports: bool = False,
    require_profiles: bool = False,
) -> dict:
    """Bind completed inventories to source identity and retained artifact bytes.

    ``path`` names one immutable attempt; log references are relative to that
    attempt and wheel references to its output directory. No referenced path
    may escape those roots, including through a symlink.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Wheel receipt must be a JSON object")
    if record.get("schema_version") != "geo-infer-wheel-receipt/1":
        raise ValueError("Unsupported wheel receipt schema")
    if (
        record.get("status") != "passed"
        or type(record.get("returncode")) is not int
        or record["returncode"] != 0
    ):
        raise ValueError("Wheel attempt did not pass")
    source, final_source = record["source"], record["final_source"]
    for value in (source, final_source):
        if not isinstance(value, dict):
            raise ValueError("Wheel source custody must be a JSON object")
        if (
            value.get("custody_complete") is not True
            or not value.get("lock_sha256")
            or not isinstance(value.get("dirty"), str)
            or not isinstance(value.get("dirty_sha256"), dict)
        ):
            raise ValueError("Wheel source custody is incomplete")
        if revision is not None and value.get("revision") != revision:
            raise ValueError("Wheel receipt belongs to another revision")
        if require_clean and value.get("dirty") != "":
            raise ValueError("Wheel receipt source is dirty")
    for key in ("revision", "dirty", "dirty_sha256", "lock_sha256"):
        if source.get(key) != final_source.get(key):
            raise ValueError("Wheel source or lock changed during execution")
    selected, executed = record["selected"], record["executed"]
    if not isinstance(selected, dict) or not isinstance(executed, dict):
        raise ValueError("Wheel execution counts must be JSON objects")
    if any(
        type(counts.get(key)) is not int or counts[key] < 0
        for counts in (selected, executed)
        for key in ("wheels", "imports", "operations")
    ):
        raise ValueError("Wheel execution counts must be nonnegative integers")
    if not modules or selected != executed or selected.get("wheels") != len(modules):
        raise ValueError("Wheel execution inventory is incomplete")
    expected_imports = (
        len(modules) if require_imports or require_profiles else selected["imports"]
    )
    expected_operations = len(profiles) if require_profiles else selected["operations"]
    if (
        selected["imports"] != expected_imports
        or selected["operations"] != expected_operations
    ):
        raise ValueError("Requested installed profiles did not execute")
    wheels = record["wheels"]
    if not isinstance(wheels, list) or any(not isinstance(row, dict) for row in wheels):
        raise ValueError("Wheel inventory must contain JSON objects")
    if sorted(row["module"] for row in wheels) != sorted(modules):
        raise ValueError(
            "Wheel module identities do not match the registered inventory"
        )
    wheel_root = path.parent.parent.parent.resolve()
    attempt_root = path.parent.resolve()

    def check_hash(root: Path, name: str, digest: str) -> None:
        relative = Path(name)
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Wheel artifact reference escapes its evidence root")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Wheel artifact hash mismatch: " + name)

    wheel_hashes = {}
    for row in wheels:
        if (
            row["status"] != "passed"
            or type(row["returncode"]) is not int
            or row["returncode"] != 0
        ):
            raise ValueError("Wheel build failed")
        artifacts = row["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("Wheel build artifacts must be a JSON object")
        if len(artifacts) != 2 or sorted(Path(name).name for name in artifacts) != [
            "stderr.log",
            "stdout.log",
        ]:
            raise ValueError("Wheel build logs are incomplete")
        for name, digest in artifacts.items():
            check_hash(attempt_root, name, digest)
        check_hash(wheel_root, row["artifact"], row["sha256"])
        wheel_hashes[Path(row["artifact"]).name.split("-")[0]] = row["sha256"]
    imports = []
    operations = []
    profile_hashes = {
        (package, profile): digest for package, profile, digest in profiles
    }
    tokens = set()
    probes = record["probes"]
    if not isinstance(probes, list) or any(
        not isinstance(probe, dict) for probe in probes
    ):
        raise ValueError("Installed probe inventory must contain JSON objects")
    for probe in probes:
        if (
            probe.get("status") != "ok"
            or type(probe.get("returncode")) is not int
            or probe["returncode"] != 0
        ):
            raise ValueError("Installed probe did not pass")
        if wheel_hashes.get(probe["package"]) != probe["wheel_sha256"]:
            raise ValueError("Installed probe belongs to another wheel")
        if not probe.get("probe_token") or probe["probe_token"] in tokens:
            raise ValueError("Installed probe completion token is missing or repeated")
        tokens.add(probe["probe_token"])
        installation = probe.get("installation_source")
        installation_digest = validate_installation_source(
            installation,
            root=Path(__file__).resolve().parents[3],
            lock_sha256=source["lock_sha256"],
        )
        if (
            probe.get("installation_source_sha256") != installation_digest
            or probe.get("installation_target") != installation["target"]
        ):
            raise ValueError("Installed child target/source binding differs")
        artifacts = probe["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("Installed probe artifacts must be a JSON object")
        if len(artifacts) != 2 or sorted(Path(name).name for name in artifacts) != [
            "stderr.log",
            "stdout.log",
        ]:
            raise ValueError("Installed probe logs are incomplete")
        completed = []
        for name, digest in artifacts.items():
            check_hash(attempt_root, name, digest)
            if Path(name).name == "stdout.log":
                for line in (
                    (attempt_root / name).read_text(encoding="utf-8").splitlines()
                ):
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if (
                        isinstance(value, dict)
                        and value.get("probe_token") == probe["probe_token"]
                    ):
                        completed.append(value)
        if len(completed) != 1 or any(
            completed[0].get(key) != probe.get(key)
            for key in (
                "package",
                "profile",
                "status",
                "version",
                "origin",
                "resources",
                "installation_source_sha256",
                "installation_target",
            )
        ):
            raise ValueError(
                "Installed probe receipt does not match its retained output"
            )
        if probe["kind"] == "import":
            if (
                probe["profile"] != "base"
                or probe["probe_sha256"] != hashlib.sha256(b"").hexdigest()
            ):
                raise ValueError("Base import probe code does not match its contract")
            imports.append(probe["package"])
        elif probe["kind"] == "operation":
            if (
                profile_hashes.get((probe["package"], probe["profile"]))
                != probe["probe_sha256"]
            ):
                raise ValueError(
                    "Operation probe code does not match the declared profile"
                )
            operations.append((probe["package"], probe["profile"]))
        else:
            raise ValueError("Unknown installed probe kind")
    expected_packages = sorted(module.lower().replace("-", "_") for module in modules)
    if expected_imports and sorted(imports) != expected_packages:
        raise ValueError(
            "Installed package identities do not match the registered inventory"
        )
    if len(imports) != expected_imports or len(operations) != expected_operations:
        raise ValueError("Installed probe counts are incomplete")
    if require_profiles and sorted(operations) != sorted(profile_hashes):
        raise ValueError("Operation profiles do not match the registered inventory")
    return record
