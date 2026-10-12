"""Installed-wheel probes must execute package code outside the source tree."""

from pathlib import Path
import importlib.util
import sys
import zipfile
import subprocess
import time
import json
import hashlib

import pytest

pytestmark = pytest.mark.unit


def _driver():
    """Load the wheel CLI module; conftest makes its sibling validator importable."""
    directory = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "geo_wheel_driver", directory / "build_package_wheels.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wheel(directory: Path, source: str) -> Path:
    """Construct a valid dependency-free wheel for an actual isolated install."""
    wheel = directory / "geo_infer_probe-0.0.1-py3-none-any.whl"
    metadata = "geo_infer_probe-0.0.1.dist-info"
    contents = {
        "geo_infer_probe/__init__.py": source,
        "geo_infer_probe/data.json": '{"value": 42}',
        metadata
        + "/METADATA": "Metadata-Version: 2.1\nName: geo-infer-probe\nVersion: 0.0.1\n",
        metadata
        + "/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    contents[metadata + "/RECORD"] = (
        "".join(name + ",,\n" for name in contents) + metadata + "/RECORD,,\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return wheel


def test_wheel_probe_rejects_broken_import(tmp_path):
    """Installed metadata alone cannot certify a package that fails to import."""
    wheel = _wheel(tmp_path, 'raise RuntimeError("broken package import")\n')
    with pytest.raises(subprocess.CalledProcessError) as error:
        _driver().install_and_verify(wheel, [sys.executable])
    assert "broken package import" in str(error.value.stderr)


def test_wheel_probe_reads_packaged_resources(tmp_path):
    """Resource lookup succeeds with only installed wheel files available."""
    wheel = _wheel(
        tmp_path,
        'from importlib.resources import files\nassert files(__package__).joinpath("data.json").is_file()\n',
    )
    evidence = tmp_path / "evidence" / "probes"
    receipts = _driver().verify_wheels([wheel], [sys.executable], evidence_dir=evidence)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["status"] == "ok" and receipt["returncode"] == 0
    assert receipt["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert "geo_infer_probe/data.json" in receipt["resources"]
    stdout = next(name for name in receipt["artifacts"] if name.endswith("stdout.log"))
    assert (
        json.loads((evidence.parent / stdout).read_text())["probe_token"]
        == receipt["probe_token"]
    )
    for name, digest in receipt["artifacts"].items():
        assert (
            hashlib.sha256((evidence.parent / name).read_bytes()).hexdigest() == digest
        )


@pytest.mark.parametrize("defect", ["missing_resource", "wrong_version", "wrong_name"])
def test_wheel_contract_checks_source_inventory(tmp_path, defect):
    """A wheel cannot certify itself by enumerating only what it happens to contain."""
    wheel = _wheel(tmp_path, "")
    module = tmp_path / "module"
    package = (
        module
        / "src"
        / ("geo_infer_other" if defect == "wrong_name" else "geo_infer_probe")
    )
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    name = "geo-infer-other" if defect == "wrong_name" else "geo-infer-probe"
    version = "9.0.0" if defect == "wrong_version" else "0.0.1"
    (module / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n'
    )
    resource = "omitted.yaml" if defect == "missing_resource" else "data.json"
    (package / resource).write_text('{"value": 42}')
    with pytest.raises(ValueError, match="metadata|omits"):
        _driver().validate_wheel_contents(wheel, module)


def test_build_missing_metadata_returns_failure(tmp_path):
    """Malformed projects produce a diagnostic instead of crashing the build driver."""
    result = _driver().build_wheel(tmp_path, tmp_path / "dist", [sys.executable])
    assert not result.ok
    assert result.wheel is None
    assert "project" in result.error


@pytest.mark.parametrize("resource", ["analysis.py", "table.csv", "__init__.py"])
def test_wheel_contract_rejects_missing_code_or_declared_data(tmp_path, resource):
    """Python code and explicitly declared data belong to the wheel contract too."""
    wheel = _wheel(tmp_path, "")
    module = tmp_path / "module"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    (module / "pyproject.toml").write_text(
        '[project]\nname = "geo-infer-probe"\nversion = "0.0.1"\n[tool.setuptools.package-data]\n"*" = ["*.csv"]\n'
    )
    if resource != "__init__.py":
        (package / "__init__.py").write_text("")
        (package / resource).write_text("value = 42")
    with pytest.raises(ValueError, match="omits|missing"):
        _driver().validate_wheel_contents(wheel, module)


def test_wheel_import_timeout_includes_stack_diagnostic(tmp_path):
    """A blocked installed import times out with the actual child stack attached."""
    wheel = _wheel(tmp_path, "import time\ntime.sleep(10)\n")
    with pytest.raises(subprocess.TimeoutExpired) as error:
        # Allow cold Windows imports to reach the deliberately blocked package
        # before the halfway-point diagnostic snapshot is taken.
        _driver().verify_wheels([wheel], [sys.executable], import_timeout=3)
    stderr = error.value.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    assert "Timeout" in stderr
    assert "geo_infer_probe/__init__.py" in stderr.replace("\\", "/")


@pytest.mark.parametrize("source", ["raise SystemExit(0)", "import os; os._exit(0)"])
def test_wheel_import_rejects_early_success_exit(tmp_path, source):
    """A zero exit cannot bypass installed provenance and resource checks."""
    wheel = _wheel(tmp_path, "print('before incomplete exit', flush=True)\n" + source)
    receipts = []
    evidence = tmp_path / "evidence" / "probes"
    with pytest.raises(ValueError, match="completion receipt"):
        _driver().verify_wheels(
            [wheel],
            [sys.executable],
            import_timeout=10,
            receipts=receipts,
            evidence_dir=evidence,
        )
    assert len(receipts) == 1 and receipts[0]["status"] == "failed"
    assert receipts[0]["returncode"] == 0
    stdout = next(
        name for name in receipts[0]["artifacts"] if name.endswith("stdout.log")
    )
    assert "before incomplete exit" in (evidence.parent / stdout).read_text()


def test_wheel_import_timeout_stops_descendants(tmp_path):
    """An installed import's spawned child cannot write after timeout cleanup."""
    started = tmp_path / "child-started"
    finished = tmp_path / "child-finished"
    child = (
        f"import pathlib,time; pathlib.Path({str(started)!r}).touch(); "
        f"time.sleep(3.0); pathlib.Path({str(finished)!r}).touch()"
    )
    wheel = _wheel(
        tmp_path,
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(10)\n",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _driver().verify_wheels([wheel], [sys.executable], import_timeout=0.5)
    # FLK-01: the same latency-tolerant pattern as the import-smoke
    # descendant test (PR #51). On a loaded runner the child's exec is not
    # bounded by any small window, so `started` can appear well after the
    # 0.5 s timeout; the old immediate assert plus a 1.3 s wait against a
    # 1.2 s child left ~0.1 s of structural margin. Poll generously; if the
    # tree kill reaped the child before exec, `started` never appears and
    # the termination property holds trivially.
    deadline = time.monotonic() + 30.0
    started_seen = started.is_file()
    while not started_seen and time.monotonic() < deadline:
        time.sleep(0.25)
        started_seen = started.is_file()
    assert not finished.exists()
    if started_seen:
        # The child did launch; a child that survived the kill reaches its
        # finished touch 3.0 s after ITS start, which is no later than
        # 3.0 s after `started` was observed. Waiting 3.5 s from the
        # observation therefore always outlasts a surviving child.
        time.sleep(3.5)
        assert not finished.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_wheel_import_rejects_nonfinite_timeout(tmp_path, timeout):
    """Import limits must remain finite even when configured through the CLI."""
    with pytest.raises(ValueError, match="finite and positive"):
        _driver().verify_wheels(
            [_wheel(tmp_path, "")], [sys.executable], import_timeout=timeout
        )


def test_fresh_build_replaces_same_name_stale_wheel(tmp_path):
    """A current build can replace its old filename and never certify old code."""
    module = tmp_path / "module"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    source = '__version__ = "0.0.1"\n'
    (package / "__init__.py").write_text(source)
    source_bytes = (package / "__init__.py").read_bytes()
    (package / "data.json").write_text('{"value": 42}')
    (module / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=61", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "geo-infer-probe"\nversion = "0.0.1"\n'
        'requires-python = ">=3.11"\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
        '[tool.setuptools.package-data]\n"*" = ["*.json"]\n'
    )
    out = tmp_path / "dist"
    out.mkdir()
    stale = _wheel(out, 'raise RuntimeError("stale build")\n')
    result = _driver().build_wheel(module, out, [sys.executable])
    assert result.ok, result.error
    assert result.wheel == stale
    with zipfile.ZipFile(result.wheel) as archive:
        assert archive.read("geo_infer_probe/__init__.py") == source_bytes
        assert archive.read("geo_infer_probe/data.json") == b'{"value": 42}'


def test_wheel_probe_preserves_shared_library_paths(tmp_path, monkeypatch):
    """Source isolation must retain configured native-library loader paths."""
    import os

    variables = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")
    values = {name: os.environ.get(name, str(tmp_path)) for name in variables}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    source = "import os\n" + "\n".join(
        f"assert os.environ.get({name!r}) == {value!r}"
        for name, value in values.items()
    )
    _driver().install_and_verify(_wheel(tmp_path, source), [sys.executable])


def _receipt_fixture(tmp_path, monkeypatch):
    """A tiny actual project and Git custody boundary for the wheel CLI."""
    driver = _driver()
    from geo_infer_test import execution

    module = tmp_path / "GEO-INFER-PROBE"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.0.1"\n')
    (package / "data.json").write_text('{"value": 42}')
    (module / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=61", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "geo-infer-probe"\nversion = "0.0.1"\n'
        'requires-python = ">=3.11"\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
        '[tool.setuptools.package-data]\n"*" = ["*.json"]\n'
    )
    (tmp_path / "uv.lock").write_bytes(
        (execution.PROJECT_ROOT / "uv.lock").read_bytes()
    )
    (tmp_path / ".gitignore").write_text("dist/\n*.egg-info/\nbuild/\n")
    for command in [
        ["git", "init", "-q"],
        ["git", "config", "core.fsmonitor", "false"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ]:
        subprocess.run(command, cwd=tmp_path, check=True)
    monkeypatch.setattr(execution, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(driver, "module_dirs", lambda: [module])
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_package_wheels.py", "--outdir", str(tmp_path / "dist"), "--verify"],
    )
    return driver, module


def test_cli_seals_actual_installed_wheel_receipt(tmp_path, monkeypatch):
    """A built/installed wheel is independently validated from retained bytes."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    from geo_infer_test.wheel_evidence import validate_wheel_receipt

    assert driver.main() == 0
    (path,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    receipt = validate_wheel_receipt(
        path,
        modules=[module.name],
        profiles=[],
        require_clean=True,
        require_imports=True,
    )
    assert (
        receipt["selected"]
        == receipt["executed"]
        == {"wheels": 1, "imports": 1, "operations": 0}
    )
    assert receipt["source"]["revision"] == receipt["final_source"]["revision"]
    assert not list(path.parent.glob("*.pending.json"))
    artifact = tmp_path / "dist" / receipt["wheels"][0]["artifact"]
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_wheel_receipt(
            path, modules=[module.name], profiles=[], require_imports=True
        )


def test_cli_retains_failed_build_and_changed_source(tmp_path, monkeypatch):
    """Changed source prevents a passing receipt even after a real clean build."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    original = driver.build_wheel

    def mutate_after_build(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.ok, result.error
        (module / "src" / "geo_infer_probe" / "__init__.py").write_text(
            '__version__ = "0.0.2"\n'
        )
        return result

    monkeypatch.setattr(driver, "build_wheel", mutate_after_build)
    assert driver.main() == 1
    (path,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed" and receipt["returncode"] == 1
    assert receipt["source"]["dirty"] == "" and receipt["final_source"]["dirty"]
    assert any("Source or lock changed" in error for error in receipt["diagnostics"])
    assert receipt["executed"]["imports"] == 1


@pytest.mark.parametrize(
    "fault",
    ["revision", "escape", "empty", "output_identity", "probe_hash", "build_output"],
)
def test_wheel_receipt_rejects_false_evidence(tmp_path, monkeypatch, fault):
    """Independent artifact and inventory defects cannot validate as acceptance."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    from geo_infer_test.wheel_evidence import validate_wheel_receipt

    assert driver.main() == 0
    (path,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    receipt = json.loads(path.read_text())
    if fault == "revision":
        receipt["final_source"]["revision"] = "0" * 40
    elif fault == "escape":
        receipt["wheels"][0]["artifact"] = "../outside.whl"
    elif fault == "empty":
        receipt["probes"] = []
    elif fault == "probe_hash":
        receipt["probes"][0]["probe_sha256"] = "0" * 64
    elif fault == "build_output":
        name = next(iter(receipt["wheels"][0]["artifacts"]))
        (path.parent / name).write_text("replaced build output")
    else:
        receipt["probes"][0]["origin"] = "/unverified/origin.py"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_wheel_receipt(
            path,
            modules=[module.name],
            profiles=[],
            require_clean=True,
            require_imports=True,
        )


def test_failed_real_build_retains_complete_output(tmp_path):
    """A real PEP517 failure retains both ends of long native diagnostics."""
    module = tmp_path / "GEO-INFER-PROBE"
    module.mkdir()
    (module / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "fixture_backend"\nbackend-path = ["."]\n'
        '[project]\nname = "geo-infer-probe"\nversion = "0.0.1"\n'
    )
    (module / "fixture_backend.py").write_text(
        "import sys\n"
        "def build_wheel(*args, **kwargs):\n"
        '    print("BUILD_STDOUT_BEGIN" + "x" * 6000 + "BUILD_STDOUT_END", flush=True)\n'
        '    print("BUILD_STDERR_BEGIN" + "y" * 6000 + "BUILD_STDERR_END", file=sys.stderr, flush=True)\n'
        '    raise RuntimeError("actual build failure")\n'
    )
    evidence = tmp_path / "evidence"
    result = _driver().build_wheel(
        module, tmp_path / "dist", [sys.executable], evidence_dir=evidence
    )
    assert not result.ok and result.returncode != 0 and result.duration_seconds > 0
    assert len(result.error) <= 2000
    retained = "".join((evidence / name).read_text() for name in result.artifacts)
    for marker in (
        "BUILD_STDOUT_BEGIN",
        "BUILD_STDOUT_END",
        "BUILD_STDERR_BEGIN",
        "BUILD_STDERR_END",
    ):
        assert marker in retained
    for name, digest in result.artifacts.items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest


def test_repeated_cli_attempts_preserve_each_wheel(tmp_path, monkeypatch):
    """A later actual build cannot overwrite an earlier attempt's proof bytes."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    from geo_infer_test.wheel_evidence import validate_wheel_receipt

    assert driver.main() == 0
    (first,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    (module / "src" / "geo_infer_probe" / "data.json").write_text('{"value": 43}')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "second fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert driver.main() == 0
    paths = list((tmp_path / "dist" / "attempts").glob("*/receipt.json"))
    assert len(paths) == 2
    receipts = [
        validate_wheel_receipt(
            path,
            modules=[module.name],
            profiles=[],
            require_clean=True,
            require_imports=True,
        )
        for path in paths
    ]
    assert len({row["source"]["revision"] for row in receipts}) == 2
    assert len({row["wheels"][0]["artifact"] for row in receipts}) == 2
    assert len({row["wheels"][0]["sha256"] for row in receipts}) == 2
    assert first.is_file()


def test_operation_code_hash_matches_live_profile(tmp_path, monkeypatch):
    """A real operation completion must name the exact declared probe body."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    from geo_infer_test.wheel_evidence import validate_wheel_receipt

    code = "assert json.loads(importlib.resources.files(package).joinpath('data.json').read_text())['value'] == 42"
    profile = driver._profiles.WheelProfile("geo_infer_probe", (), code)
    monkeypatch.setattr(driver, "REQUIRED_PROFILES", (profile,))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_package_wheels.py",
            "--outdir",
            str(tmp_path / "dist"),
            "--verify-extras",
        ],
    )
    assert driver.main() == 0
    (path,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    expected = [(profile.package, "base", hashlib.sha256(code.encode()).hexdigest())]
    validate_wheel_receipt(
        path, modules=[module.name], profiles=expected, require_profiles=True
    )
    receipt = json.loads(path.read_text())
    operation = next(row for row in receipt["probes"] if row["kind"] == "operation")
    operation["probe_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="probe code"):
        validate_wheel_receipt(
            path, modules=[module.name], profiles=expected, require_profiles=True
        )


@pytest.mark.parametrize(
    "fault",
    [
        "root",
        "source",
        "counts",
        "wheels",
        "wheel_artifacts",
        "probes",
        "probe_artifacts",
    ],
)
def test_nested_receipt_shapes_fail_with_controlled_verdict(
    tmp_path, monkeypatch, fault
):
    """Malformed JSON structure fails as evidence, without attribute crashes."""
    driver, module = _receipt_fixture(tmp_path, monkeypatch)
    from geo_infer_test.wheel_evidence import validate_wheel_receipt

    assert driver.main() == 0
    (path,) = (tmp_path / "dist" / "attempts").glob("*/receipt.json")
    record = json.loads(path.read_text())
    if fault == "root":
        record = []
    elif fault == "source":
        record["source"] = []
    elif fault == "counts":
        record["selected"] = []
    elif fault == "wheels":
        record["wheels"] = [[]]
    elif fault == "wheel_artifacts":
        record["wheels"][0]["artifacts"] = []
    elif fault == "probes":
        record["probes"] = [[]]
    else:
        record["probes"][0]["artifacts"] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_wheel_receipt(
            path, modules=[module.name], profiles=[], require_imports=True
        )
