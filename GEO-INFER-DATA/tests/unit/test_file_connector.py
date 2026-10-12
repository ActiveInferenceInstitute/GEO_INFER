"""
Tests for FileConnector and StreamingFileConnector in geo_infer_data.connectors.file.
"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from geo_infer_data.connectors.file import FileConnector, StreamingFileConnector


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# FileConnector
# ---------------------------------------------------------------------------


class TestFileConnector:
    def test_init_creates_base_path(self, tmp_path):
        base = tmp_path / "test_base"
        connector = FileConnector(base_path=str(base))
        assert base.exists()
        assert connector.base_path == base

    def test_list_files_empty(self, tmp_path):
        connector = FileConnector(base_path=str(tmp_path))
        files = connector.list_files("*.geojson")
        assert files == []

    def test_list_files_finds_matching(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "map.geojson").write_text("{}")
        (tmp_path / "readme.txt").write_text("hello")
        connector = FileConnector(base_path=str(tmp_path))
        csv_files = connector.list_files("*.csv")
        assert len(csv_files) == 1
        assert csv_files[0].name == "data.csv"

    def test_list_files_with_file_types_filter(self, tmp_path):
        (tmp_path / "a.csv").write_text("x\n1\n")
        (tmp_path / "b.geojson").write_text("{}")
        (tmp_path / "c.txt").write_text("hi")
        connector = FileConnector(base_path=str(tmp_path))
        files = connector.list_files("*", file_types=["csv", "geojson"])
        assert len(files) == 2

    def test_list_files_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.csv").write_text("a\n1\n")
        (tmp_path / "top.csv").write_text("b\n2\n")
        connector = FileConnector(base_path=str(tmp_path))
        files = connector.list_files("*.csv", recursive=True)
        assert len(files) == 2

    def test_read_geospatial_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        connector = FileConnector(base_path=str(tmp_path))
        df = _run(connector.read_geospatial(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_read_geospatial_nonexistent_raises(self, tmp_path):
        connector = FileConnector(base_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            _run(connector.read_geospatial(tmp_path / "missing.csv"))

    def test_write_csv(self, tmp_path):
        connector = FileConnector(base_path=str(tmp_path))
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result_path = _run(connector.write_geospatial(df, tmp_path / "out.csv"))
        assert Path(result_path).exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 3

    def test_write_parquet(self, tmp_path):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.fail("pyarrow not installed")
        connector = FileConnector(base_path=str(tmp_path))
        df = pd.DataFrame({"a": range(10)})
        result_path = _run(connector.write_geospatial(df, tmp_path / "out.parquet"))
        assert Path(result_path).exists()

    def test_scan_directory(self, tmp_path):
        (tmp_path / "data.csv").write_text("a\n1\n")
        (tmp_path / "map.geojson").write_text("{}")
        (tmp_path / "image.tif").write_bytes(b"\x00" * 100)
        connector = FileConnector(base_path=str(tmp_path))
        stats = _run(connector.scan_directory())
        assert stats["total_files"] == 3
        assert stats["geospatial_files"] >= 2
        assert "by_format" in stats

    def test_scan_nonexistent_directory(self, tmp_path):
        connector = FileConnector(base_path=str(tmp_path))
        stats = _run(connector.scan_directory("/nonexistent/path"))
        assert "error" in stats

    def test_compress_and_extract_zip(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        archive = tmp_path / "archive.zip"
        connector = FileConnector(base_path=str(tmp_path))
        result = _run(connector.compress_files([f1, f2], archive, compression="zip"))
        assert Path(result).exists()

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        extracted = _run(connector.extract_archive(archive, extract_dir))
        assert len(extracted) == 2

    def test_extract_nonexistent_archive_raises(self, tmp_path):
        connector = FileConnector(base_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            _run(connector.extract_archive(tmp_path / "missing.zip"))


# ---------------------------------------------------------------------------
# StreamingFileConnector
# ---------------------------------------------------------------------------


class TestStreamingFileConnector:
    def test_init_default_chunk_size(self):
        connector = StreamingFileConnector()
        assert connector.chunk_size == 10000

    def test_init_custom_chunk_size(self):
        connector = StreamingFileConnector(chunk_size=500)
        assert connector.chunk_size == 500

    def test_read_csv_streaming(self, tmp_path):
        csv_path = tmp_path / "big.csv"
        df = pd.DataFrame({"value": range(50)})
        df.to_csv(csv_path, index=False)

        connector = StreamingFileConnector(chunk_size=20)

        async def collect_chunks():
            chunks = []
            async for chunk in connector.read_csv_streaming(str(csv_path)):
                chunks.append(chunk)
            return chunks

        chunks = _run(collect_chunks())
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 50
        assert len(chunks) == 3  # 20 + 20 + 10

    def test_write_csv_streaming(self, tmp_path):
        out_path = tmp_path / "streamed.csv"
        connector = StreamingFileConnector()

        def data_gen():
            for i in range(3):
                yield pd.DataFrame({"col": range(i * 10, (i + 1) * 10)})

        result = _run(connector.write_csv_streaming(data_gen(), str(out_path)))
        assert Path(result).exists()
        df = pd.read_csv(result)
        assert len(df) == 30
