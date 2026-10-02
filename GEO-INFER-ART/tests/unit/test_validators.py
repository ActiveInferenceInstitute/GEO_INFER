#!/usr/bin/env python
"""
Unit tests for input validators, the custom algorithm framework, the
performance optimizer helpers, and the animation fallback helper.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from geo_infer_art.utils.validators import (
    validate_file_path,
    validate_geospatial_data,
)
from geo_infer_art.core.generation.custom_algorithms import CustomAlgorithmFramework
from geo_infer_art.core.generation.performance_optimizer import (
    PerformanceOptimizer,
    cache_result,
    time_execution,
)
from geo_infer_art.utils.animation import save_animation_with_fallback


def _sample_algorithm(data, params, width, height):
    """Deterministic grid generator supplied by trusted Python callers."""
    offset = float(params.get("offset", 0.0))
    return [[(x + y) * 0.01 + offset for x in range(width)] for y in range(height)]


class TestValidators(unittest.TestCase):
    """Boundary tests for the geo_infer_art.utils.validators helpers."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name

    def _tmp_file(self, name):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write("x")
        return path

    def test_validate_file_path_missing_raises(self):
        missing = os.path.join(self.tmpdir, "nope.geojson")
        with self.assertRaises(FileNotFoundError):
            validate_file_path(missing)

    def test_validate_file_path_wrong_extension_raises(self):
        path = self._tmp_file("data.txt")
        with self.assertRaises(ValueError):
            validate_file_path(path, extensions=[".geojson", ".json"])

    def test_validate_file_path_allowed_extension_ok(self):
        path = self._tmp_file("data.geojson")
        validate_file_path(path, extensions=[".geojson", ".json"])

    def test_validate_geospatial_data_plain_list_raises(self):
        with self.assertRaises(ValueError):
            validate_geospatial_data([[1, 2], [3, 4]])

    def test_validate_geospatial_data_1d_array_raises(self):
        with self.assertRaises(ValueError):
            validate_geospatial_data(np.arange(6))

    def test_validate_geospatial_data_two_channels_raises(self):
        with self.assertRaises(ValueError):
            validate_geospatial_data(np.zeros((4, 4, 2)))

    def test_validate_geospatial_data_valid_2d_ok(self):
        validate_geospatial_data(np.zeros((4, 4)))


class TestCustomAlgorithmFramework(unittest.TestCase):
    """Tests for registration, execution, and save/load of custom algorithms."""

    def test_register_duplicate_name_raises(self):
        framework = CustomAlgorithmFramework()
        framework.register_algorithm("sample", _sample_algorithm)
        with self.assertRaises(ValueError):
            framework.register_algorithm("sample", _sample_algorithm)

    def test_register_non_callable_raises(self):
        framework = CustomAlgorithmFramework()
        with self.assertRaises(ValueError):
            framework.register_algorithm("sample", "not-a-function")

    def test_register_missing_required_params_raises(self):
        framework = CustomAlgorithmFramework()

        def bad_algorithm(data, params):
            return data

        with self.assertRaises(ValueError):
            framework.register_algorithm("sample", bad_algorithm)

    def test_execute_unknown_name_raises(self):
        framework = CustomAlgorithmFramework()
        with self.assertRaises(ValueError):
            framework.execute_algorithm("nope", None, width=4, height=3)

    def test_execute_matches_direct_call(self):
        framework = CustomAlgorithmFramework()
        framework.register_algorithm("sample", _sample_algorithm)
        result = framework.execute_algorithm(
            "sample", None, width=4, height=3, offset=1.5
        )
        expected = _sample_algorithm(None, {"offset": 1.5}, 4, 3)
        self.assertEqual(result, expected)

    def test_save_load_round_trip(self):
        registry = {"sample_grid": _sample_algorithm}
        source_framework = CustomAlgorithmFramework(registry)
        source_framework.register_algorithm(
            "sample",
            _sample_algorithm,
            description="sample algo",
            parameters={"offset": "grid offset"},
            example_usage="run sample",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "algorithms.json")
            source_framework.save_algorithms_to_file(path)
            with open(path) as file:
                payload = json.load(file)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                payload["algorithms"]["sample"]["registry_key"], "sample_grid"
            )
            self.assertNotIn("source", payload["algorithms"]["sample"])
            target = CustomAlgorithmFramework(registry)
            target.load_algorithms_from_file(path)
        self.assertEqual(
            target.get_algorithm_info("sample")["description"], "sample algo"
        )
        self.assertEqual(
            target.execute_algorithm("sample", None, 4, 3, offset=1.5),
            _sample_algorithm(None, {"offset": 1.5}, 4, 3),
        )

    def test_unlisted_callable_cannot_be_saved(self):
        framework = CustomAlgorithmFramework()
        framework.register_algorithm("sample", _sample_algorithm)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "algorithms.json")
            with self.assertRaisesRegex(ValueError, "no trusted registry key"):
                framework.save_algorithms_to_file(path)
            self.assertFalse(os.path.exists(path))

    def test_legacy_source_is_rejected_without_execution(self):
        import builtins

        payload = {
            "sample": {
                "metadata": {},
                "function_name": "sample",
                "source": "import builtins; builtins._art_exec_marker = True",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.json")
            with open(path, "w") as file:
                json.dump(payload, file)
            framework = CustomAlgorithmFramework()
            with self.assertRaisesRegex(ValueError, "schema_version 1"):
                framework.load_algorithms_from_file(path)
        self.assertFalse(hasattr(builtins, "_art_exec_marker"))
        self.assertEqual(framework.list_algorithms(), [])

    def test_invalid_entry_leaves_registry_unchanged(self):
        metadata = {"description": "", "parameters": {}, "example_usage": ""}
        payload = {
            "schema_version": 1,
            "algorithms": {
                "good": {"registry_key": "sample_grid", "metadata": metadata},
                "bad": {"registry_key": "os:system", "metadata": metadata},
            },
        }
        framework = CustomAlgorithmFramework({"sample_grid": _sample_algorithm})
        framework.register_algorithm("existing", _sample_algorithm)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mixed.json")
            with open(path, "w") as file:
                json.dump(payload, file)
            with self.assertRaisesRegex(ValueError, "Unknown registry key"):
                framework.load_algorithms_from_file(path)
        self.assertEqual(framework.list_algorithms(), ["existing"])


class TestPerformanceOptimizer(unittest.TestCase):
    """Tests for the caching and parallel helpers around art generation."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = os.path.join(tmp.name, "cache")
        self.optimizer = PerformanceOptimizer(cache_dir=self.cache_dir)

    def test_cached_execution_caches_numpy_results(self):
        calls = []

        def make_array(n):
            calls.append(n)
            return np.arange(n)

        first = self.optimizer.cached_execution(make_array, args=(6,))
        second = self.optimizer.cached_execution(make_array, args=(6,))
        np.testing.assert_array_equal(first, np.arange(6))
        np.testing.assert_array_equal(second, first)
        self.assertEqual(len(calls), 1)

        third = self.optimizer.cached_execution(make_array, args=(8,))
        np.testing.assert_array_equal(third, np.arange(8))
        self.assertEqual(len(calls), 2)

    def test_cached_execution_passes_non_cacheable_through(self):
        calls = []

        def make_text():
            calls.append(1)
            return "txt"

        first = self.optimizer.cached_execution(make_text)
        second = self.optimizer.cached_execution(make_text)
        self.assertEqual(first, "txt")
        self.assertEqual(second, "txt")
        self.assertEqual(len(calls), 2)

    def test_parallel_execution_maps_failures_to_none(self):
        def flaky(ok):
            if not ok:
                raise RuntimeError("boom")
            return ok

        results = self.optimizer.parallel_execution(
            flaky, [{"ok": True}, {"ok": False}]
        )
        self.assertEqual(results, [True, None])

    def test_cache_result_decorator_caches_numpy_results(self):
        calls = []

        @cache_result(self.optimizer)
        def make_array(n):
            calls.append(n)
            return np.arange(n)

        first = make_array(6)
        second = make_array(6)
        np.testing.assert_array_equal(first, np.arange(6))
        np.testing.assert_array_equal(second, first)
        self.assertEqual(len(calls), 1)

    def test_time_execution_returns_result_and_elapsed(self):
        @time_execution
        def slow_add(a, b):
            time.sleep(0.02)
            return a + b

        result, elapsed = slow_add(2, 3)
        self.assertEqual(result, 5)
        self.assertGreaterEqual(elapsed, 0.02)


class TestSaveAnimationWithFallback(unittest.TestCase):
    """Tests for the animation export fallback helper."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name
        self.addCleanup(plt.close, "all")

    def _make_animation(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        anim = FuncAnimation(fig, lambda i: ax.plot([0, i + 1]), frames=2)
        return anim

    def test_gif_target_saves_directly(self):
        anim = self._make_animation()
        output_path = os.path.join(self.tmpdir, "out.gif")
        returned = save_animation_with_fallback(anim, output_path, 2)
        self.assertEqual(returned, output_path)
        with Image.open(output_path) as img:
            self.assertEqual(img.format, "GIF")
            self.assertGreaterEqual(getattr(img, "n_frames", 1), 2)

    def test_creates_missing_output_directory(self):
        anim = self._make_animation()
        output_path = os.path.join(self.tmpdir, "sub", "out.gif")
        returned = save_animation_with_fallback(anim, output_path, 2)
        self.assertTrue(os.path.exists(returned))

    def test_ffmpeg_failure_falls_back_to_gif(self):
        calls = []

        def fake_save(self, path, writer=None, fps=None):
            calls.append((path, writer))
            if writer == "ffmpeg":
                raise RuntimeError("ffmpeg unavailable")
            with open(path, "wb") as f:
                f.write(b"GIF89a")

        anim = self._make_animation()
        mp4_path = os.path.join(self.tmpdir, "out.mp4")
        with (
            mock.patch("matplotlib.animation.FuncAnimation.save", fake_save),
            self.assertLogs("geo_infer_art.utils.animation", level="WARNING"),
        ):
            returned = save_animation_with_fallback(anim, mp4_path, 2)
        # The mocked save never starts a real draw, so mark the animation as
        # rendered to keep its __del__ from emitting an unraisable warning.
        anim._draw_was_started = True

        self.assertEqual(returned, os.path.join(self.tmpdir, "out.gif"))
        self.assertTrue(os.path.exists(returned))
        self.assertEqual(calls[0][1], "ffmpeg")
        self.assertEqual(calls[1][1], "pillow")


if __name__ == "__main__":
    unittest.main()
