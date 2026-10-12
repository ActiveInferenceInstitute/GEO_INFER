#!/usr/bin/env python3
"""Test import of CascadianAgriculturalH3Backend"""


def test_import_cascadian_backend():
    """CascadianAgriculturalH3Backend must be importable."""
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

    assert CascadianAgriculturalH3Backend is not None
