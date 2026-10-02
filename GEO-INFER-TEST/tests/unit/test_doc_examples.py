"""Documentation gates reject phantom acceptance and unsafe manifest paths."""

import json
from pathlib import Path

import pytest

from geo_infer_test.doc_examples import example_code, load_manifest, verify_page
from geo_infer_test import execution


@pytest.mark.parametrize("entries", [[], ["../outside.md"], ["/outside.md"], [4]])
def test_invalid_manifest_rejected(tmp_path, entries):
    manifest = tmp_path / "examples.json"
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        load_manifest(tmp_path, manifest)


def test_duplicate_pages_rejected(tmp_path):
    (tmp_path / "page.md").write_text("```python\nassert True\n```\n")
    manifest = tmp_path / "examples.json"
    manifest.write_text('["page.md", "page.md"]')
    with pytest.raises(ValueError, match="duplicates"):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    "text",
    [
        "# No examples",
        "```python\nprint(42)\n```\n",
        "Illustrative example notice\n```python\nassert True\n```\n",
    ],
)
def test_unverifiable_pages_rejected(tmp_path, text):
    page = tmp_path / "page.md"
    page.write_text(text)
    with pytest.raises(ValueError):
        example_code(page)


@pytest.mark.parametrize(
    "code, passed",
    [
        ("assert 1 + 1 == 2", True),
        ("assert 1 + 1 == 3", False),
        ("assert True\nraise SystemExit(0)", False),
        ("import os\nassert True\nos._exit(0)", False),
    ],
)
def test_real_example_execution_requires_completion(
    tmp_path, monkeypatch, code, passed
):
    monkeypatch.setattr(execution, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(execution, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        execution,
        "runtime_receipt",
        lambda **kwargs: {"revision": "fixture", "custody_complete": True},
    )
    page = tmp_path / "page.md"
    page.write_text(f"```python\n{code}\n```\n")
    result = verify_page(page, timeout=5)
    assert result.success is passed
    receipt = json.loads(Path(result.receipt).read_text())
    assert receipt["status"] == ("PASS" if passed else "FAIL")
    assert (Path(result.receipt).parent / "stdout.log").is_file()
