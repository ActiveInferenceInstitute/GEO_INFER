"""California source custody uses the installed owner resources."""

from __future__ import annotations

import hashlib
from importlib.resources import files

from geo_infer_place.locations.california import contracts


def test_execution_binding_does_not_require_a_checkout(monkeypatch, tmp_path):
    # A wheel or alternate loader has no five-parent monorepo ancestry.
    monkeypatch.setattr(contracts, "__file__", str(tmp_path / "contracts.py"))
    binding = contracts.execution_binding()

    expected_packages = {"numpy", "scipy", "h3", "pydantic", "pyproj", "shapely"}
    assert set(binding.packages) == expected_packages
    assert all(binding.packages.values())
    assert binding.owners
    assert len(binding.owners) == 14
    for key, digest in binding.owners.items():
        module, source, package, *relative = key.split("/")
        assert source == "src"
        assert module == f"GEO-INFER-{package.removeprefix('geo_infer_').upper()}"
        payload = files(package).joinpath(*relative).read_bytes()
        assert digest == hashlib.sha256(payload).hexdigest()

    assert (
        "GEO-INFER-ACT/src/geo_infer_act/core/resource_allocation.py" in binding.owners
    )
    assert (
        "GEO-INFER-PLACE/src/geo_infer_place/core/_regional_download_worker.py"
        in binding.owners
    )
