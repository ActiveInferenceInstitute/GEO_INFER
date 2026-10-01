"""
Encryption utilities for geospatial data.

This module provides cryptographic functions for securing sensitive
geospatial data both at rest and in transit.
"""

import base64
import os
import json
from typing import Any, cast
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)

import logging

logger = logging.getLogger(__name__)
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class GeospatialEncryption:
    """Provides encryption methods for geospatial data."""

    def __init__(self, key: bytes | None = None):
        """
        Initialize the encryptor with a key.

        Args:
            key: Encryption key (if None, a new one will be generated)
        """
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key

        self.cipher = Fernet(self.key)
        self.salt: bytes | None = None

    @classmethod
    def from_password(
        cls, password: str, salt: bytes | None = None
    ) -> "GeospatialEncryption":
        """
        Create an encryptor using a password-derived key.

        Args:
            password: Password to derive key from
            salt: Salt for key derivation (if None, a new one will be generated)

        Returns:
            GeospatialEncryption instance
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        instance = cls(key)
        instance.salt = salt

        return instance

    def get_key(self) -> bytes:
        """Get the encryption key."""
        return self.key

    def encrypt_text(self, text: str) -> bytes:
        """
        Encrypt a text string.

        Args:
            text: Text to encrypt

        Returns:
            Encrypted data as bytes
        """
        return self.cipher.encrypt(text.encode())

    def decrypt_text(self, encrypted_data: bytes) -> str:
        """
        Decrypt encrypted text data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted text
        """
        return self.cipher.decrypt(encrypted_data).decode()

    def encrypt_json(self, data: dict) -> bytes:
        """
        Encrypt a dictionary as JSON.

        Args:
            data: Dictionary to encrypt

        Returns:
            Encrypted data as bytes
        """
        json_str = json.dumps(data)
        return self.encrypt_text(json_str)

    def decrypt_json(self, encrypted_data: bytes) -> dict:
        """
        Decrypt JSON data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted dictionary
        """
        json_str = self.decrypt_text(encrypted_data)
        return cast(dict[Any, Any], json.loads(json_str))

    def encrypt_coordinates(self, lat: float, lon: float) -> str:
        """
        Encrypt latitude and longitude.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Encrypted coordinates as a base64 string
        """
        coord_str = f"{lat},{lon}"
        encrypted = self.encrypt_text(coord_str)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_coordinates(self, encrypted_coords: str) -> tuple[float, float]:
        """
        Decrypt coordinates.

        Args:
            encrypted_coords: Encrypted coordinates string

        Returns:
            Tuple of (latitude, longitude)
        """
        encrypted = base64.urlsafe_b64decode(encrypted_coords)
        coord_str = self.decrypt_text(encrypted)
        lat_str, lon_str = coord_str.split(",")
        return float(lat_str), float(lon_str)

    def encrypt_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        sensitive_columns: list[str] | None = None,
        encrypt_coordinates: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Encrypt sensitive columns in a GeoDataFrame.

        Args:
            gdf: GeoDataFrame to encrypt
            sensitive_columns: List of column names to encrypt (if None, all non-geometry columns are encrypted)
            encrypt_coordinates: Whether to encrypt the geometry coordinates

        Returns:
            GeoDataFrame with encrypted data
        """
        result = gdf.copy()

        # Encrypt specified columns
        if sensitive_columns is None:
            sensitive_columns = [col for col in result.columns if col != "geometry"]

        for col in sensitive_columns:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: (
                        base64.urlsafe_b64encode(self.encrypt_text(str(x))).decode()
                        if pd.notna(x)
                        else None
                    )
                )

        # Encrypt coordinates if requested
        if encrypt_coordinates:
            if "encrypted_geometry" not in result.columns:
                result["encrypted_geometry"] = None

            for idx, row in result.iterrows():
                geom = row.geometry
                if isinstance(geom, Point):
                    result.at[idx, "encrypted_geometry"] = self.encrypt_coordinates(
                        geom.y, geom.x
                    )

        return result

    def decrypt_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        encrypted_columns: list[str],
        geometry_col: str = "encrypted_geometry",
    ) -> gpd.GeoDataFrame:
        """
        Decrypt columns in a GeoDataFrame.

        Args:
            gdf: GeoDataFrame with encrypted data
            encrypted_columns: List of encrypted column names
            geometry_col: Name of the column with encrypted geometries

        Returns:
            GeoDataFrame with decrypted data
        """
        result = gdf.copy()

        # Decrypt specified columns
        for col in encrypted_columns:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: (
                        self.decrypt_text(base64.urlsafe_b64decode(x))
                        if pd.notna(x)
                        else None
                    )
                )

        # Decrypt geometries if present. Corrupted/undecryptable rows are not
        # silently discarded: each failure is logged and its row index is
        # collected into ``result.attrs["decryption_failures"]`` so callers
        # can detect and handle the data loss explicitly.
        decryption_failures: list[int] = []
        if geometry_col in result.columns:
            for idx, row in result.iterrows():
                if pd.notna(row[geometry_col]):
                    try:
                        lat, lon = self.decrypt_coordinates(row[geometry_col])
                        result.at[idx, "geometry"] = Point(lon, lat)
                    except Exception as e:
                        logger.warning(
                            "Failed to decrypt geometry at row %s: %s", idx, e
                        )
                        decryption_failures.append(int(idx))

        if decryption_failures:
            result.attrs["decryption_failures"] = decryption_failures

        return result


class AsymmetricEncryption:
    """Provides asymmetric encryption for secure data sharing."""

    def __init__(
        self,
        private_key: rsa.RSAPrivateKey | None = None,
        public_key: rsa.RSAPublicKey | None = None,
    ):
        """
        Initialize with optional existing keys.

        Args:
            private_key: RSA private key
            public_key: RSA public key
        """
        self.private_key = private_key
        self.public_key = public_key

    @classmethod
    def generate_keys(cls, key_size: int = 2048) -> "AsymmetricEncryption":
        """
        Generate a new key pair.

        Args:
            key_size: Size of the RSA key in bits

        Returns:
            AsymmetricEncryption instance with new keys
        """
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        public_key = private_key.public_key()

        return cls(private_key, public_key)

    def export_private_key(self) -> bytes:
        """
        Export the private key in PEM format.

        Returns:
            PEM encoded private key
        """
        if self.private_key is None:
            raise ValueError("No private key available")

        return self.private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )

    def export_public_key(self) -> bytes:
        """
        Export the public key in PEM format.

        Returns:
            PEM encoded public key
        """
        if self.public_key is None:
            raise ValueError("No public key available")

        return self.public_key.public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        )

    @classmethod
    def from_pem(
        cls,
        private_key_pem: bytes | None = None,
        public_key_pem: bytes | None = None,
    ) -> "AsymmetricEncryption":
        """
        Create an instance from PEM encoded keys.

        Args:
            private_key_pem: PEM encoded private key
            public_key_pem: PEM encoded public key

        Returns:
            AsymmetricEncryption instance
        """
        private_key = None
        public_key = None

        if private_key_pem:
            private_key = cast(
                rsa.RSAPrivateKey, load_pem_private_key(private_key_pem, password=None)
            )

        if public_key_pem:
            public_key = cast(rsa.RSAPublicKey, load_pem_public_key(public_key_pem))

        return cls(private_key, public_key)

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using the public key.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        if self.public_key is None:
            raise ValueError("No public key available for encryption")

        # RSA can only encrypt limited data size, so use a hybrid approach
        # Generate a symmetric key and encrypt it with RSA
        symmetric_key = Fernet.generate_key()

        # Encrypt the symmetric key with RSA
        encrypted_key = self.public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Use the symmetric key to encrypt the actual data
        cipher = Fernet(symmetric_key)
        encrypted_data = cipher.encrypt(data)

        # Combine the encrypted key and data with a delimiter
        return (
            base64.urlsafe_b64encode(encrypted_key)
            + b"|"
            + base64.urlsafe_b64encode(encrypted_data)
        )

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the private key.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data
        """
        if self.private_key is None:
            raise ValueError("No private key available for decryption")

        # Split the encrypted key and data
        encrypted_key_b64, encrypted_data_b64 = encrypted_data.split(b"|")
        encrypted_key = base64.urlsafe_b64decode(encrypted_key_b64)
        encrypted_data = base64.urlsafe_b64decode(encrypted_data_b64)

        # Decrypt the symmetric key
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Use the symmetric key to decrypt the data
        cipher = Fernet(symmetric_key)
        return cipher.decrypt(encrypted_data)

    def encrypt_text(self, text: str) -> bytes:
        """
        Encrypt a text string.

        Args:
            text: Text to encrypt

        Returns:
            Encrypted data
        """
        return self.encrypt(text.encode())

    def decrypt_text(self, encrypted_data: bytes) -> str:
        """
        Decrypt text data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted text
        """
        return self.decrypt(encrypted_data).decode()

    def encrypt_json(self, data: dict) -> bytes:
        """
        Encrypt a dictionary as JSON.

        Args:
            data: Dictionary to encrypt

        Returns:
            Encrypted data
        """
        return self.encrypt_text(json.dumps(data))

    def decrypt_json(self, encrypted_data: bytes) -> dict:
        """
        Decrypt JSON data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted dictionary
        """
        return cast(dict[Any, Any], json.loads(self.decrypt_text(encrypted_data)))
