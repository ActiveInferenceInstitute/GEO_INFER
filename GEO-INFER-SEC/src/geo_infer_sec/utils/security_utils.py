"""
Security utilities for GEO-INFER framework.

This module provides utility functions for security and privacy protection
in geospatial data processing and analysis.
"""

import hashlib
import hmac
import secrets
import base64
import os
from typing import Any, cast
from dataclasses import dataclass
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# Characters removed by SecurityUtils.strip_dangerous_chars(). See that
# method's warning: this list is a cosmetic normalisation aid, not
# injection defense.
_DANGEROUS_CHARS = ("<", ">", '"', "'", "&", ";", "|", "`", "$", "(", ")")


@dataclass
class SecurityConfig:
    """Configuration for security utilities."""

    # Encryption parameters
    key_length: int = 32
    salt_length: int = 16
    iterations: int = 100000

    # Token signing secret. A stable secret is required so that
    # generate_secure_token() and validate_token() agree on the HMAC key.
    # When unset it falls back to GEO_INFER_TOKEN_SECRET, then to a
    # per-process random key (valid only within one process lifetime).
    token_secret: str | None = None

    # Privacy parameters
    k_anonymity: int = 5
    l_diversity: int = 3
    t_closeness: float = 0.1

    # Access control
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # seconds

    # Audit parameters
    enable_audit_logging: bool = True
    audit_retention_days: int = 90


class SecurityUtils:
    """
    Utility class for security and privacy operations.

    Provides methods for data encryption, privacy protection,
    access control, and audit logging.
    """

    def __init__(self, config: SecurityConfig | None = None):
        """
        Initialize security utilities.

        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.failed_attempts: dict[str, tuple[int, datetime]] = {}
        self.audit_log: list[dict[str, Any]] = []
        # Stable HMAC key for token sign/verify. Prefer an explicit secret,
        # then the env var, then a per-process random key.
        secret = self.config.token_secret or os.getenv("GEO_INFER_TOKEN_SECRET")
        if secret:
            self._token_key = hashlib.sha256(secret.encode("utf-8")).digest()
        else:
            self._token_key = secrets.token_bytes(self.config.key_length)
            logger.warning(
                "No token_secret configured (SecurityConfig.token_secret or "
                "GEO_INFER_TOKEN_SECRET); tokens are only valid within this process."
            )

    def generate_secure_key(self, length: int | None = None) -> bytes:
        """
        Generate a cryptographically secure random key.

        Args:
            length: Key length in bytes (uses config default if None)

        Returns:
            Random key as bytes
        """
        key_len = length or self.config.key_length
        return secrets.token_bytes(key_len)

    def hash_password(
        self, password: str, salt: bytes | None = None
    ) -> tuple[bytes, bytes]:
        """
        Hash a password using PBKDF2.

        Args:
            password: Password to hash
            salt: Salt for hashing (generated if None)

        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(self.config.salt_length)

        # Use PBKDF2 for password hashing
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.config.iterations
        )

        return hash_obj, salt

    def verify_password(self, password: str, stored_hash: bytes, salt: bytes) -> bool:
        """
        Verify a password against stored hash.

        Args:
            password: Password to verify
            stored_hash: Stored password hash
            salt: Salt used for hashing

        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, stored_hash)

    def encrypt_data(self, data: str | bytes, key: bytes) -> tuple[bytes, bytes]:
        """
        Encrypt data using AES-256-CBC.

        Args:
            data: Data to encrypt
            key: Encryption key (must be 32 bytes for AES-256)

        Returns:
            Tuple of (encrypted_data, iv)

        Raises:
            ValueError: If key length is not 32 bytes
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Ensure key is 32 bytes for AES-256
        if len(key) != 32:
            # Derive 32-byte key from provided key using SHA-256
            key = hashlib.sha256(key).digest()

        # Generate initialization vector
        iv = secrets.token_bytes(16)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Pad data to block size (16 bytes for AES)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data)
        padded_data += padder.finalize()

        # Encrypt
        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted, iv

    def decrypt_data(self, encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """
        Decrypt data using AES-256-CBC.

        Args:
            encrypted_data: Encrypted data
            key: Decryption key (must be 32 bytes for AES-256)
            iv: Initialization vector (must be 16 bytes)

        Returns:
            Decrypted data

        Raises:
            ValueError: If key or IV length is incorrect
        """
        # Ensure key is 32 bytes for AES-256
        if len(key) != 32:
            # Derive 32-byte key from provided key using SHA-256
            key = hashlib.sha256(key).digest()

        if len(iv) != 16:
            raise ValueError("IV must be 16 bytes")

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data)
        data += unpadder.finalize()

        return data

    def anonymize_spatial_data(
        self,
        data: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        precision: int = 2,
    ) -> pd.DataFrame:
        """
        Anonymize spatial data by reducing precision.

        Args:
            data: Input DataFrame
            lat_col: Latitude column name
            lon_col: Longitude column name
            precision: Decimal precision to keep

        Returns:
            Anonymized DataFrame
        """
        anonymized = data.copy()

        if lat_col in anonymized.columns:
            anonymized[lat_col] = anonymized[lat_col].round(precision)

        if lon_col in anonymized.columns:
            anonymized[lon_col] = anonymized[lon_col].round(precision)

        return anonymized

    def apply_k_anonymity(
        self,
        data: pd.DataFrame,
        sensitive_cols: list[str],
        quasi_identifiers: list[str],
    ) -> pd.DataFrame:
        """
        Apply k-anonymity to protect sensitive data.

        Args:
            data: Input DataFrame
            sensitive_cols: Columns containing sensitive information
            quasi_identifiers: Columns that could identify individuals

        Returns:
            K-anonymized DataFrame
        """
        k = self.config.k_anonymity
        anonymized = data.copy()

        # Group by quasi-identifiers
        groups = anonymized.groupby(quasi_identifiers)

        # For each group, ensure at least k records
        for _, group in groups:
            if len(group) < k:
                # Suppress or generalize values
                for col in quasi_identifiers:
                    anonymized.loc[group.index, col] = "*"

        return anonymized

    def add_noise_to_numerical(
        self,
        data: pd.DataFrame,
        columns: list[str],
        noise_level: float = 0.1,
        rng: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to numerical columns for privacy protection.

        Args:
            data: Input DataFrame
            columns: Columns to add noise to
            noise_level: Standard deviation of noise as fraction of data std
            rng: Integer seed or ``np.random.Generator`` for reproducible noise

        Returns:
            DataFrame with added noise
        """
        generator = np.random.default_rng(rng)
        noisy_data = data.copy()

        for col in columns:
            if col in noisy_data.columns and pd.api.types.is_numeric_dtype(
                noisy_data[col]
            ):
                std = noisy_data[col].std()
                noise = generator.normal(0, std * noise_level, len(noisy_data))
                noisy_data[col] = noisy_data[col] + noise

        return noisy_data

    def check_access_control(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission to perform action on resource.

        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed

        Returns:
            True if access is allowed, False otherwise
        """
        # Check for account lockout
        if self._is_account_locked(user_id):
            self._log_audit_event(user_id, resource, action, "DENIED", "Account locked")
            return False

        # Simple access control (in practice, use proper RBAC/ABAC)
        allowed_resources = self._get_user_permissions(user_id)

        if resource in allowed_resources:
            self._log_audit_event(user_id, resource, action, "ALLOWED")
            return True
        else:
            self._log_audit_event(
                user_id, resource, action, "DENIED", "Insufficient permissions"
            )
            return False

    def _is_account_locked(self, user_id: str) -> bool:
        """Check if user account is locked due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False

        attempts, last_attempt = self.failed_attempts[user_id]

        if attempts >= self.config.max_failed_attempts:
            lockout_until = last_attempt + timedelta(
                seconds=self.config.lockout_duration
            )
            if datetime.now() < lockout_until:
                return True
            else:
                # Reset failed attempts after lockout period
                del self.failed_attempts[user_id]

        return False

    def _get_user_permissions(self, user_id: str) -> list[str]:
        """Get list of resources user has permission to access."""
        # Simplified permission system
        # In practice, this would query a database or external system
        permissions = {
            "admin": ["*"],  # Admin has access to everything
            "analyst": ["geo_infer_space", "geo_infer_place", "geo_infer_iot"],
            "viewer": ["geo_infer_space"],
        }

        # Extract role from user_id (simplified)
        if "admin" in user_id.lower():
            role = "admin"
        elif "analyst" in user_id.lower():
            role = "analyst"
        else:
            role = "viewer"

        return permissions.get(role, [])

    def record_failed_attempt(self, user_id: str) -> None:
        """Record a failed authentication attempt."""
        now = datetime.now()

        if user_id in self.failed_attempts:
            attempts, _ = self.failed_attempts[user_id]
            self.failed_attempts[user_id] = (attempts + 1, now)
        else:
            self.failed_attempts[user_id] = (1, now)

    def _log_audit_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        reason: str | None = None,
    ) -> None:
        """Log an audit event."""
        if not self.config.enable_audit_logging:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "reason": reason,
            "event_id": str(uuid.uuid4()),
        }

        self.audit_log.append(event)
        logger.info(f"Audit: {user_id} {action} {resource} - {result}")

    def get_audit_log(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get audit log entries with optional filtering.

        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            user_id: User ID for filtering

        Returns:
            List of audit log entries
        """
        filtered_log = self.audit_log.copy()

        if start_time:
            filtered_log = [
                e
                for e in filtered_log
                if datetime.fromisoformat(e["timestamp"]) >= start_time
            ]

        if end_time:
            filtered_log = [
                e
                for e in filtered_log
                if datetime.fromisoformat(e["timestamp"]) <= end_time
            ]

        if user_id:
            filtered_log = [e for e in filtered_log if e["user_id"] == user_id]

        return filtered_log

    def cleanup_audit_log(self) -> None:
        """Remove old audit log entries based on retention policy."""
        if not self.config.enable_audit_logging:
            return

        cutoff_date = datetime.now() - timedelta(days=self.config.audit_retention_days)

        self.audit_log = [
            e
            for e in self.audit_log
            if datetime.fromisoformat(e["timestamp"]) > cutoff_date
        ]

    def generate_secure_token(self, user_id: str, expiration_hours: int = 24) -> str:
        """
        Generate a secure token for user authentication.

        Args:
            user_id: User identifier
            expiration_hours: Token expiration time in hours

        Returns:
            Secure token string
        """
        payload = {
            "user_id": user_id,
            "exp": datetime.now() + timedelta(hours=expiration_hours),
            "nonce": secrets.token_hex(16),
        }

        # Encode payload
        payload_str = json.dumps(payload, default=str)
        payload_bytes = payload_str.encode("utf-8")

        # Create signature
        key = self._token_key
        signature = hmac.new(key, payload_bytes, hashlib.sha256).digest()

        # Base64 each part separately (urlsafe alphabet has no '.'), so the
        # '.' separator can never collide with payload or signature bytes.
        token = (
            base64.urlsafe_b64encode(payload_bytes)
            + b"."
            + base64.urlsafe_b64encode(signature)
        )
        return token.decode("utf-8")

    def validate_token(self, token: str) -> str | None:
        """
        Validate a secure token and return user ID if valid.

        Args:
            token: Token to validate

        Returns:
            User ID if token is valid, None otherwise
        """
        try:
            # Decode token: <urlsafe_b64(payload)>.<urlsafe_b64(signature)>
            p_b64, sig_b64 = token.encode("utf-8").split(b".", 1)
            payload_str = base64.urlsafe_b64decode(p_b64)
            signature = base64.urlsafe_b64decode(sig_b64)

            # Verify signature
            key = self._token_key
            expected_signature = hmac.new(key, payload_str, hashlib.sha256).digest()

            if not hmac.compare_digest(signature, expected_signature):
                return None

            # Parse payload
            payload = json.loads(payload_str.decode("utf-8"))

            # Check expiration
            exp_str = payload["exp"]
            if isinstance(exp_str, str):
                exp = datetime.fromisoformat(exp_str)
            else:
                exp = exp_str

            if datetime.now() > exp:
                return None

            return cast(str, payload["user_id"])

        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None

    def strip_dangerous_chars(self, input_data: str) -> str:
        """
        Strip characters that are inconvenient in downstream display or
        shell contexts.

        .. warning::
            This is NOT injection defense. Removing ``<``, ``"``, ``'``,
            ``;``, ``|`` and friends from a string does not make that
            string safe to interpolate into SQL, HTML, or shell commands;
            encodings and context-specific payloads survive character
            stripping. Injection is prevented exclusively by parameterised
            queries, context-aware output encoding, and argument-vector
            execution at the point of use. Use this only as a cosmetic
            normalisation helper.

        Args:
            input_data: Input string to normalise.

        Returns:
            The string with the characters in ``_DANGEROUS_CHARS`` removed.
        """
        for char in _DANGEROUS_CHARS:
            input_data = input_data.replace(char, "")
        return input_data

    def validate_file_upload(
        self, file_path: str, allowed_extensions: list[str], max_size_mb: int = 10
    ) -> tuple[bool, str]:
        """
        Validate file upload for security.

        Args:
            file_path: Path to uploaded file
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        import os

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            return False, f"File extension {file_ext} not allowed"

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return (
                False,
                f"File size {file_size_mb:.2f}MB exceeds maximum {max_size_mb}MB",
            )

        return True, "File upload valid"


# Convenience functions
def create_security_utils(config: SecurityConfig | None = None) -> SecurityUtils:
    """Create a new SecurityUtils instance."""
    return SecurityUtils(config)


def hash_password_simple(password: str) -> str:
    """Simple password hashing function."""
    salt = secrets.token_bytes(16)
    hash_obj, salt = SecurityUtils().hash_password(password, salt)
    return base64.b64encode(hash_obj + salt).decode("utf-8")


def verify_password_simple(password: str, stored_hash: str) -> bool:
    """Simple password verification function."""
    try:
        hash_data = base64.b64decode(stored_hash.encode("utf-8"))
        hash_obj = hash_data[:-16]
        salt = hash_data[-16:]
        return SecurityUtils().verify_password(password, hash_obj, salt)
    except Exception:
        return False


def generate_secure_token(length: int = 32) -> str:
    """Generate a URL-safe random token of the given byte length."""
    return secrets.token_urlsafe(length)


_PII_COLUMN_HINTS = (
    "email",
    "phone",
    "ssn",
    "social",
    "birthdate",
    "birth",
    "address",
    "first_name",
    "last_name",
    "full_name",
    "passport",
    "driver",
    "license",
    "account",
    "card",
    "iban",
    "tax",
    "medical",
    "pin",
)


def check_pii_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose names suggest they hold personally identifiable data."""
    lowered = {str(col).lower(): str(col) for col in df.columns}
    return [lowered[keyword] for keyword in _PII_COLUMN_HINTS if keyword in lowered]


def validate_spatial_bounds(gdf: Any) -> bool:
    """Return True if a GeoDataFrame's bounds fall within WGS84 extents."""
    import geopandas as gpd

    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return False
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    in_lon = -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
    in_lat = -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
    return bool(in_lon and in_lat)


def detect_outliers(gdf: Any) -> pd.DataFrame:
    """Flag rows whose centroids fall outside the inter-quartile range."""
    import geopandas as gpd

    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        result = pd.DataFrame(index=range(len(gdf)))
        result["is_outlier"] = False
        return result

    centroids = gdf.geometry.centroid
    q25 = centroids.x.quantile(0.25)
    q75 = centroids.y.quantile(0.75)
    iqr = q75 - q25
    lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    result = gdf.copy()
    result["is_outlier"] = (centroids.y < lower) | (centroids.y > upper)
    return result
