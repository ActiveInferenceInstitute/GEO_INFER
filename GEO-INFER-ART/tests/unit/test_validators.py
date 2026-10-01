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
    """Deterministic grid generator (pure builtins; exec'd from saved source)."""
    offset = float(params.get("offset", 0.0))
    return [[(x + y) * 0.01 + offset for x in range(width)] for y in range(height)]


# Source-text fixture for saved-file entries (the framework exec's this
# string; a literal copy, so tests never parse this file's own text).
_SAMPLE_ALGORITHM_SOURCE = '''\
def _sample_algorithm(data, params, width, height):
    """Deterministic grid generator (pure builtins; exec'd from saved source)."""
    offset = float(params.get("offset", 0.0))
    return [
        [(x + y) * 0.01 + offset for x in range(width)] for y in range(height)
    ]
'''


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
        source_framework = CustomAlgorithmFramework()
        source_framework.register_algorithm(
            name="sample",
            algorithm_function=_sample_algorithm,
            description="sample algo",
            parameters={"offset": "grid offset"},
            example_usage="framework.execute_algorithm('sample', None, 4, 3)",
        )

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        saved_path = os.path.join(tmp.name, "algorithms.json")
        source_framework.save_algorithms_to_file(saved_path)

        target_framework = CustomAlgorithmFramework()
        target_framework.load_algorithms_from_file(saved_path)

        self.assertIn("sample", target_framework.list_algorithms())
        info = target_framework.get_algorithm_info("sample")
        self.assertEqual(info["description"], "sample algo")
        self.assertEqual(info["parameters"], {"offset": "grid offset"})
        self.assertEqual(
            info["example_usage"],
            "framework.execute_algorithm('sample', None, 4, 3)",
        )

        loaded_result = target_framework.execute_algorithm(
            "sample", None, width=4, height=3, offset=1.5
        )
        self.assertEqual(loaded_result, _sample_algorithm(None, {"offset": 1.5}, 4, 3))

    def test_load_rejects_entry_without_function_name(self):
        """Files without ``function_name`` are rejected before any exec."""
        payload = {
            "sample": {
                "metadata": {
                    "description": "entry without function_name",
                    "parameters": {},
                    "example_usage": "",
                },
                # Executing this source would register a marker; rejection
                # must happen before the source runs.
                "source": _SAMPLE_ALGORITHM_SOURCE
                + "\nimport builtins\nbuiltins._geo_infer_art_exec_marker = True\n",
            }
        }

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "no_function_name.json")
        with open(path, "w") as f:
            json.dump(payload, f)

        framework = CustomAlgorithmFramework()
        with self.assertRaisesRegex(
            ValueError, "entry 'sample' is missing function_name"
        ):
            framework.load_algorithms_from_file(path)
        self.assertNotIn("sample", framework.list_algorithms())
        import builtins

        self.assertFalse(hasattr(builtins, "_geo_infer_art_exec_marker"))

    def test_load_skips_entry_whose_function_is_missing(self):
        payload = {
            "sample": {
                "metadata": {"description": "", "parameters": {}, "example_usage": ""},
                "source": _SAMPLE_ALGORITHM_SOURCE,
                "function_name": "not_defined_here",
            }
        }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "wrong_name.json")
        with open(path, "w") as f:
            json.dump(payload, f)

        framework = CustomAlgorithmFramework()
        with self.assertLogs(
            "geo_infer_art.core.generation.custom_algorithms", level="WARNING"
        ) as captured:
            framework.load_algorithms_from_file(path)
        self.assertNotIn("sample", framework.list_algorithms())
        self.assertIn("not_defined_here", "\n".join(captured.output))

    def test_load_skips_bad_entries_with_warning(self):
        source = _SAMPLE_ALGORITHM_SOURCE
        payload = {
            "good": {
                "metadata": {
                    "description": "good entry",
                    "parameters": {},
                    "example_usage": "",
                },
                "source": source,
                "function_name": "_sample_algorithm",
            },
            "bad": {
                "metadata": {
                    "description": "bad entry",
                    "parameters": {},
                    "example_usage": "",
                },
                "source": "def broken(:\n    pass",
                "function_name": "broken",
            },
        }

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mixed_path = os.path.join(tmp.name, "mixed.json")
        with open(mixed_path, "w") as f:
            json.dump(payload, f)

        framework = CustomAlgorithmFramework()
        with self.assertLogs(
            "geo_infer_art.core.generation.custom_algorithms", level="WARNING"
        ) as captured:
            framework.load_algorithms_from_file(mixed_path)

        self.assertIn("good", framework.list_algorithms())
        self.assertNotIn("bad", framework.list_algorithms())
        warning_text = "\n".join(captured.output)
        self.assertIn("Could not load algorithm 'bad'", warning_text)


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
