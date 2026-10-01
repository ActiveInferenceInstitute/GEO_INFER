"""Shared upload helpers for GEO-INFER-PEP API endpoints."""

import logging
from pathlib import Path
import tempfile

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


async def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Persist an uploaded file to a temporary path.

    The caller is responsible for unlinking the returned path (typically in a
    ``finally`` block). Raises ``HTTPException(500)`` with a generic message if
    the file cannot be written; the underlying error is only logged server-side.
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=upload_file.filename
        ) as tmp:
            contents = await upload_file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)
    except Exception:
        logger.exception("Could not save uploaded file to a temporary path")
        raise HTTPException(
            status_code=500, detail="Could not save uploaded file"
        ) from None
    finally:
        await upload_file.close()
    return tmp_path
