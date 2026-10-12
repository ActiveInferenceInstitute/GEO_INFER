"""Strict optional Bayeux imports with one verified upstream docstring exception."""

from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
from types import ModuleType
import warnings


# jaxopt 0.8.3 ships a non-raw BoxOSQP docstring containing \mu and \phi.
# CPython 3.11/3.12 reports its first invalid escape at these distinct lines.
_OSQP_SHA256 = "1abe8d507f4880f74fce5a989cf5948915b655424372dca573efd83039c29ce8"


def _import_bayeux() -> ModuleType:
    """Import the actual backend without retrying or replacing import failures.

    The exception applies only to the identified dependency source compilation;
    runtime warnings use a dotted module name and remain subject to caller policy.
    Changed dependency source is not covered by this compatibility exception.
    """
    with warnings.catch_warnings():
        try:
            dependency = distribution("jaxopt")
        except PackageNotFoundError:
            dependency = None
        if dependency is not None and dependency.version == "0.8.3":
            source = Path(dependency.locate_file("jaxopt/_src/osqp.py"))
            if sha256(source.read_bytes()).hexdigest() == _OSQP_SHA256:
                # Compiler warnings identify their module by filename without
                # .py; filtering jaxopt._src.osqp would miss the cold import.
                module = "^" + re.escape(str(source.with_suffix(""))) + "$"
                for category, line in ((DeprecationWarning, 299), (SyntaxWarning, 333)):
                    warnings.filterwarnings(
                        "ignore",
                        message=r"^invalid escape sequence '\\m'$",
                        category=category,
                        module=module,
                        lineno=line,
                    )
        return import_module("bayeux")
