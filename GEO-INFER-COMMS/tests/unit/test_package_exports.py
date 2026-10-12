"""Package-root export and singleton tests (M3-05, GS19-45, GS19-57).

Covers the unified-interface re-exports (collaboration, streaming, spatial
routing), the CollaborationManager lifecycle wiring, and the double-checked
lock around the global communication-system singleton.
"""

import threading

import geo_infer_comms
from geo_infer_comms import (
    AdvancedSpatialRouter,
    CollaborationManager,
    GeospatialCommunicationSystem,
    StreamManager,
)

UNIFIED_INTERFACE_NAMES = (
    "CollaborationManager",
    "StreamManager",
    "AdvancedSpatialRouter",
    "GeospatialCollaborationCoordinator",
    "CollaborationAnalytics",
)


def test_unified_interface_exports():
    """The documented unified interface exposes the collaboration engines."""
    for name in UNIFIED_INTERFACE_NAMES:
        assert hasattr(geo_infer_comms, name), name
        assert name in geo_infer_comms.__all__, name


def test_unified_interface_probe_import():
    """The supplement's acceptance probe: direct import of the three engines."""
    from geo_infer_comms import (
        CollaborationManager as cm,
        StreamManager as sm,
        AdvancedSpatialRouter as asr,
    )

    assert cm is CollaborationManager
    assert sm is StreamManager
    assert asr is AdvancedSpatialRouter


def test_system_wires_collaboration_manager():
    """GeospatialCommunicationSystem owns a live CollaborationManager."""
    system = GeospatialCommunicationSystem()
    assert isinstance(system.collaboration_manager, CollaborationManager)


def test_first_call_race_yields_single_instance(monkeypatch):
    """Concurrent first access must not leak multiple system instances."""
    monkeypatch.setattr(geo_infer_comms, "_global_system", None)

    results = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        results.append(geo_infer_comms.get_communication_system())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(results) == 4
    assert len({id(instance) for instance in results}) == 1
