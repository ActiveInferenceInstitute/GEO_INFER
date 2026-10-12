#!/usr/bin/env python3
"""
Verification Script for GEO-INFER-SPACE Installation.

This script verifies that the package is correctly installed and that
all backends are functioning with REAL methods (no mocks).
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_h3_backend():
    """Verify H3 backend uses real methods."""
    logger.info("Verifying H3 Backend...")
    try:
        from geo_infer_space.backends.h3 import H3Backend

        backend = H3Backend()

        # 1. Check availability
        if not backend.is_available():
            logger.error("❌ H3 Backend is not available (H3 library missing?)")
            return False

        # 2. Check real methods
        # Real H3 resolution 9 cell for SF
        lat, lng = 37.7749, -122.4194
        cell = backend.latlng_to_cell(lat, lng, 9)

        # Verify it looks like a real H3 cell (15 chars, hex)
        if len(cell) != 15:
            logger.error(f"❌ H3 latlng_to_cell returned suspect value: {cell}")
            return False

        # Verify new methods
        res = backend.get_cell_resolution(cell)
        if res != 9:
            logger.error(f"❌ H3 get_cell_resolution returned incorrect value: {res}")
            return False

        area = backend.get_cell_area(cell)
        if not (0.09 < area < 0.11):  # Approx area for res 9
            logger.error(f"❌ H3 get_cell_area returned suspect value: {area}")
            return False

        logger.info("✅ H3 Backend verification passed (Real H3 methods active)")
        return True

    except ImportError as e:
        logger.error(f"❌ Failed to import H3 backend: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ H3 Backend verification failed: {e}")
        return False


def verify_srai_backend():
    """Verify SRAI backend uses real methods."""
    logger.info("Verifying SRAI Backend...")
    try:
        from geo_infer_space.backends.srai import SraiBackend

        backend = SraiBackend()

        # 1. Check availability (might be missing, which is valid for optional dep)
        if not backend.is_available():
            logger.warning(
                "⚠️ SRAI Backend is not available (SRAI library not installed)"
            )
            logger.info("   This is acceptable if SRAI is optional.")
            return True  # Not a failure, just missing optional dep

        # 2. Check real methods if available
        lat, lng = 37.7749, -122.4194
        cell = backend.latlng_to_cell(lat, lng, 9)

        # SRAI (using H3 regionalizer) should match H3
        if len(cell) != 15:
            logger.error(f"❌ SRAI latlng_to_cell returned suspect value: {cell}")
            return False

        logger.info("✅ SRAI Backend verification passed (Real SRAI methods active)")
        return True

    except ImportError as e:
        logger.error(f"❌ Failed to import SRAI backend: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ SRAI Backend verification failed: {e}")
        return False


def verify_dispatcher():
    """Verify Dispatcher works correctly."""
    logger.info("Verifying Spatial Dispatcher...")
    try:
        from geo_infer_space import get_backend_dispatcher
        from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

        dispatcher = get_backend_dispatcher()

        # Check backend registration
        backends = dispatcher.get_available_backends()
        logger.info(f"   Registered backends: {backends}")

        if "h3" not in backends:
            logger.error("❌ H3 backend not registered in dispatcher")
            return False

        # Real call through Interface (which uses dispatcher)
        indexing = SpatialIndexingInterface()
        lat, lng = 37.7749, -122.4194
        cell = indexing.latlng_to_cell(lat, lng, 9)

        if len(cell) != 15:
            logger.error(f"❌ Interface latlng_to_cell returned suspect value: {cell}")
            return False

        logger.info("✅ Dispatcher/Interface verification passed")
        return True

    except Exception as e:
        logger.error(f"❌ Dispatcher verification failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting Verification Process...")

    h3_ok = verify_h3_backend()
    srai_ok = verify_srai_backend()
    disp_ok = verify_dispatcher()

    if h3_ok and srai_ok and disp_ok:
        logger.info("\n🎉 ALL VERIFICATIONS PASSED: System is using REAL methods.")
        sys.exit(0)
    else:
        logger.error("\n❌ VERIFICATION FAILED: Some checks failed.")
        sys.exit(1)
