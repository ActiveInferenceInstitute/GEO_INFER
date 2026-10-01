"""Regression tests for GS-085: SPACE H3 predicate helpers must distinguish
invalid cells from unexpected failures (expected classes silent-False,
unexpected classes logged at debug, never silently swallowed).

The helpers run against the installed h3-py. An over-long hexadecimal string
makes h3-py raise ``OverflowError``, which is outside the expected
``TypeError``/``ValueError`` classes and exercises the logged branch.
"""

import unittest

import h3

import geo_infer_space.backends.h3.operations as ops

LOGGER_NAME = "geo_infer_space.backends.h3.operations"
OVERFLOW_INDEX = "f" * 40
CELL = h3.latlng_to_cell(37.7749, -122.4194, 9)
NEIGHBOR = sorted(set(h3.grid_disk(CELL, 1)) - {CELL})[0]


class TestOperationsPredicateHelpers(unittest.TestCase):
    def test_invalid_cell_returns_false(self) -> None:
        self.assertFalse(ops.is_valid_cell("not-a-cell"))

    def test_valid_cell_returns_true(self) -> None:
        self.assertTrue(ops.is_valid_cell(CELL))

    def test_unexpected_error_is_logged(self) -> None:
        with self.assertRaises(OverflowError):
            h3.is_valid_cell(OVERFLOW_INDEX)
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cap:
            self.assertFalse(ops.is_valid_cell(OVERFLOW_INDEX))
        self.assertIn("Unexpected error", cap.output[0])

    def test_neighbor_unexpected_error_is_logged(self) -> None:
        with self.assertRaises(OverflowError):
            h3.are_neighbor_cells(CELL, OVERFLOW_INDEX)
        with self.assertLogs(LOGGER_NAME, level="DEBUG") as cap:
            self.assertFalse(ops.are_neighbor_cells(CELL, OVERFLOW_INDEX))
        self.assertIn("Unexpected error", cap.output[0])

    def test_neighbor_bad_type_returns_false(self) -> None:
        with self.assertRaises(TypeError):
            h3.are_neighbor_cells(CELL, None)
        with self.assertNoLogs(LOGGER_NAME, level="DEBUG"):
            self.assertFalse(ops.are_neighbor_cells(CELL, None))  # type: ignore[arg-type]

    def test_true_neighbors(self) -> None:
        self.assertTrue(ops.are_neighbor_cells(CELL, NEIGHBOR))


if __name__ == "__main__":
    unittest.main()
