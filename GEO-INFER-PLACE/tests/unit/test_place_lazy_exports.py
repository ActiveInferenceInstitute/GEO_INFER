from __future__ import annotations

import json
import subprocess
import sys


def test_small_place_imports_do_not_load_forecasting_or_ml() -> None:
    """Import lightweight PLACE APIs in a fresh process without heavy backends."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json;import geo_infer_place as p;import geo_infer_place.core as c;from geo_infer_place.core.regional_layers import _download;print(json.dumps({'forecasting':'geo_infer_time.core.forecasting' in sys.modules,'sklearn':'sklearn' in sys.modules,'exports':p.__all__,'listed':all(name in dir(p) for name in p.__all__),'core_listed':all(name in dir(c) for name in c.__all__)}))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert not observed["forecasting"] and not observed["sklearn"]
    assert (
        len(observed["exports"]) == 36
        and observed["listed"]
        and observed["core_listed"]
    )


def test_public_export_identity_and_unknown_attribute_errors() -> None:
    """Preserve public class identity and reject unknown lazy export names."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import geo_infer_place as p;import geo_infer_place.core as c;from geo_infer_place.core.base_module import BaseAnalysisModule;assert p.BaseAnalysisModule is c.BaseAnalysisModule is BaseAnalysisModule;assert p.BaseAnalysisModule is p.BaseAnalysisModule;assert p.latlng_to_cell(38,-121,5);assert p.cell_to_latlng(p.latlng_to_cell(38,-121,5));\nfor module in (p,c):\n try: module.does_not_exist\n except AttributeError: continue\n raise AssertionError('missing attribute incorrectly resolved')\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
