"""Portable docstring guards retain strict unrelated warning and failure semantics."""

from hashlib import sha256
from types import ModuleType, SimpleNamespace
import warnings

import pytest

from geo_infer_act.utils import bayeux_backend as backend


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    path = tmp_path / "site-packages" / "jaxopt" / "_src" / "osqp.py"
    path.parent.mkdir(parents=True)
    # One multiline docstring: Python3.11 attributes the escape to its
    # opening line299; Python3.12 attributes it to the actual escape line333.
    source = (
        "# filler\n" * 298 + '"""\n' + "docstring\n" * 33 + r"\mu" + '\n"""\n'
    ).encode()
    monkeypatch.setattr(backend, "_OSQP_SHA256", sha256(source).hexdigest())
    path.write_bytes(source)
    metadata = SimpleNamespace(version="0.8.3", locate_file=lambda _: path)
    monkeypatch.setattr(backend, "distribution", lambda name: metadata)
    return path, source, metadata


def test_guarded_docstring_compile_is_accepted_only_inside_backend_import(
    dependency, monkeypatch
):
    path, source, _ = dependency
    module = ModuleType("bayeux")
    calls = []

    def import_backend(name):
        calls.append(name)
        compile(source, str(path), "exec")
        return module

    monkeypatch.setattr(backend, "import_module", import_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        assert backend._import_bayeux() is module
        assert warnings.filters == before
        with pytest.raises(SyntaxError, match="invalid escape"):
            compile(source, str(path), "exec")
    assert calls == ["bayeux"]


@pytest.mark.parametrize(
    "category,line", [(DeprecationWarning, 299), (SyntaxWarning, 333)]
)
@pytest.mark.parametrize(
    "corruption", ["message", "filename", "line", "runtime_module", "category"]
)
def test_unrelated_warnings_remain_fatal(
    dependency, monkeypatch, category, line, corruption
):
    path, _, _ = dependency
    message = "invalid escape sequence '\\m'"
    filename = str(path)
    module = str(path.with_suffix(""))
    if corruption == "message":
        message = "invalid escape sequence '\\q'"
    elif corruption == "filename":
        filename = str(path.with_name("another.py"))
        module = filename[:-3]
    elif corruption == "line":
        line += 1
    elif corruption == "runtime_module":
        module = "jaxopt._src.osqp"
    elif corruption == "category":
        category = UserWarning

    def import_backend(_):
        warnings.warn_explicit(
            message, category, filename=filename, lineno=line, module=module
        )

    monkeypatch.setattr(backend, "import_module", import_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        with pytest.raises(category, match="invalid escape"):
            backend._import_bayeux()
        assert warnings.filters == before


@pytest.mark.parametrize("corruption", ["version", "source"])
def test_changed_dependency_does_not_receive_exception(
    dependency, monkeypatch, corruption
):
    path, source, metadata = dependency
    if corruption == "version":
        metadata.version = "0.8.4"
    else:
        path.write_bytes(source + b"\n# dependency changed\n")
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _: compile(path.read_bytes(), str(path), "exec"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SyntaxError, match="invalid escape"):
            backend._import_bayeux()


@pytest.mark.parametrize(
    "error",
    [ImportError("real backend unavailable"), RuntimeError("initialization failed")],
)
def test_original_import_failure_propagates_once_and_restores_filters(
    dependency, monkeypatch, error
):
    calls = []

    def import_backend(name):
        calls.append(name)
        raise error

    monkeypatch.setattr(backend, "import_module", import_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        with pytest.raises(type(error)) as caught:
            backend._import_bayeux()
        assert caught.value is error
        assert warnings.filters == before
    assert calls == ["bayeux"]


def test_missing_jaxopt_retains_actual_backend_import_error(monkeypatch):
    def no_dependency(_):
        raise backend.PackageNotFoundError("jaxopt")

    error = ImportError("No module named jaxopt")

    def import_backend(_):
        raise error

    monkeypatch.setattr(backend, "distribution", no_dependency)
    monkeypatch.setattr(backend, "import_module", import_backend)
    with pytest.raises(ImportError) as caught:
        backend._import_bayeux()
    assert caught.value is error


def test_legacy_xla_deprecation_is_not_suppressed(dependency, monkeypatch):
    message = "jax.interpreters.xla.pytype_aval_mappings is deprecated."

    def import_backend(_):
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    monkeypatch.setattr(backend, "import_module", import_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        with pytest.raises(DeprecationWarning, match="pytype_aval_mappings"):
            backend._import_bayeux()
        assert warnings.filters == before
