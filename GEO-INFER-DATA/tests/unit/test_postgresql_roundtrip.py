"""Actual SQL routing regressions; SQLite exercises SQL without claiming PostGIS acceptance."""

from datetime import datetime, UTC

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from geo_infer_data.core.storage import PostgreSQLBackend
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


@pytest.fixture
def backend(tmp_path):
    backend = PostgreSQLBackend(
        {
            "host": "localhost",
            "port": 5432,
            "user": "test",
            "password": "test",
            "database": "test",
        }
    )
    backend.connection_string = f"sqlite:///{tmp_path / 'storage.sqlite'}"
    return backend


@pytest.fixture
def metadata():
    return DatasetMetadata(
        title="SQL roundtrip",
        spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
        temporal=TemporalExtent(
            start=datetime(2023, 1, 1, tzinfo=UTC), end=datetime(2023, 1, 2, tzinfo=UTC)
        ),
        lineage=DataLineage(source="test", process="roundtrip", created_by="test"),
    )


@pytest.mark.asyncio
async def test_generic_public_roundtrip_and_delete(backend, metadata):
    payloads = [{"value": 7}, [1, "two"], None, "text", 3]
    ids = [await backend.store(payload, metadata) for payload in payloads]
    assert len(set(ids)) == len(ids)
    assert [await backend.retrieve(data_id, {}) for data_id in ids] == payloads
    with pytest.raises(ValueError, match="do not support table queries"):
        await backend.retrieve(ids[0], {"spatial": [0, 0, 1, 1]})
    assert await backend.delete(ids[0])
    assert not await backend.delete(ids[0])
    with pytest.raises(FileNotFoundError):
        await backend.retrieve(ids[0], {})
    assert await backend.retrieve(ids[1], {}) == payloads[1]


@pytest.mark.asyncio
async def test_tabular_public_roundtrip_no_overwrite(backend, metadata):
    first = pd.DataFrame({"value": [1, 2]})
    second = pd.DataFrame({"value": [8]})
    first_id = await backend.store(first, metadata)
    second_id = await backend.store(second, metadata)
    assert first_id != second_id
    pd.testing.assert_frame_equal(await backend.retrieve(first_id, {}), first)
    pd.testing.assert_frame_equal(await backend.retrieve(second_id, {}), second)
    with pytest.raises(ValueError, match="already exists"):
        await backend._store_dataframe(second, first_id, metadata)
    pd.testing.assert_frame_equal(await backend.retrieve(first_id, {}), first)
    assert await backend.delete(first_id)
    assert not await backend.delete(first_id)
    pd.testing.assert_frame_equal(await backend.retrieve(second_id, {}), second)


@pytest.mark.asyncio
async def test_generic_decode_rejects_corruption(backend, metadata):
    data_id = await backend.store({"value": 7}, metadata)
    engine = create_engine(backend.connection_string)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE generic_data_store SET payload = :payload WHERE data_id = :data_id"
                ),
                {"payload": "not valid JSON", "data_id": data_id},
            )
        assert inspect(engine).has_table("generic_data_store")
    finally:
        engine.dispose()
    with pytest.raises(ValueError, match="neither JSON"):
        await backend.retrieve(data_id, {})


@pytest.mark.asyncio
async def test_geospatial_write_uses_transaction_without_in_memory_index(
    backend, metadata, monkeypatch
):
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame({"value": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    observed = []

    def external_writer(self, name, connection, **kwargs):
        assert connection.in_transaction()
        assert kwargs == {"if_exists": "fail", "index": False}
        observed.append(name)
        pd.DataFrame({"value": self["value"]}).to_sql(name, connection, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_postgis", external_writer)
    data_id = await backend.store(frame, metadata)
    assert observed == [f"dataset_{data_id}"]
    assert backend.spatial_indexer.indexes == {}
    assert (await backend.retrieve(data_id, {}))["value"].tolist() == [1]
