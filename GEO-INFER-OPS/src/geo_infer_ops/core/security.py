"""
Security management for GEO-INFER-OPS.
"""

import os
import jwt
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import load_pem_x509_certificate
from cryptography.x509.oid import NameOID
import base64

from geo_infer_ops.core.config import get_config
from geo_infer_ops.core.logging import get_logger

logger = get_logger(__name__)


class SecurityManager:
    """Manages security operations for GEO-INFER-OPS."""

    def __init__(self) -> None:
        """Initialize security manager."""
        self.config = get_config()
        self._load_keys(allow_missing=True)

    def _load_keys(self, allow_missing: bool = False) -> None:
        """Load encryption and signing keys."""
        try:
            # Load TLS certificate and key
            if self.config.security.tls.enabled:
                cert_file = self.config.security.tls.cert_file
                key_file = self.config.security.tls.key_file
                if (
                    cert_file
                    and key_file
                    and Path(cert_file).exists()
                    and Path(key_file).exists()
                ):
                    with open(cert_file, "rb") as f:
                        self.cert = load_pem_x509_certificate(f.read())
                    with open(key_file, "rb") as f:
                        self.key = serialization.load_pem_private_key(
                            f.read(), password=None
                        )
                elif not allow_missing:
                    raise FileNotFoundError("TLS certificate or key file is missing")

            # Load JWT secret
            if self.config.security.auth.enabled:
                jwt_secret = self.config.security.auth.jwt_secret
                if jwt_secret is not None:
                    self.jwt_secret = jwt_secret.encode()

            # Initialize encryption
            self.fernet = Fernet(Fernet.generate_key())

            logger.info("security_keys_loaded")
        except Exception as e:
            logger.error("security_keys_load_failed", error=str(e))
            raise

    def generate_tls_certificate(
        self, common_name: str, organization: str, country: str, days_valid: int = 365
    ) -> dict[str, str]:
        """
        Generate a self-signed TLS certificate.

        Args:
            common_name: Common name for the certificate
            organization: Organization name
            country: Country code
            days_valid: Number of days the certificate is valid

        Returns:
            Dict containing certificate and key paths
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            # Generate public key
            public_key = private_key.public_key()

            # Generate certificate
            subject = issuer = cryptography.x509.Name(
                [
                    cryptography.x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                    cryptography.x509.NameAttribute(
                        NameOID.ORGANIZATION_NAME, organization
                    ),
                    cryptography.x509.NameAttribute(NameOID.COUNTRY_NAME, country),
                ]
            )

            cert = (
                cryptography.x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(public_key)
                .serial_number(cryptography.x509.random_serial_number())
                .not_valid_before(datetime.now(UTC))
                .not_valid_after(datetime.now(UTC) + timedelta(days=days_valid))
                .add_extension(
                    cryptography.x509.SubjectAlternativeName(
                        [
                            cryptography.x509.DNSName(common_name),
                            cryptography.x509.DNSName(f"*.{common_name}"),
                        ]
                    ),
                    critical=False,
                )
                .sign(private_key, hashes.SHA256())
            )

            # Save certificate and key
            cert_path = Path(self.config.security.tls.cert_file or "")
            key_path = Path(self.config.security.tls.key_file or "")

            cert_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))

            with open(key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )

            logger.info("tls_certificate_generated", common_name=common_name)
            return {"cert_file": str(cert_path), "key_file": str(key_path)}
        except Exception as e:
            logger.error("tls_certificate_generation_failed", error=str(e))
            raise

    def generate_csr(
        self, common_name: str, organization: str, country: str, key_size: int = 2048
    ) -> str:
        """
        Generate a Certificate Signing Request (CSR).

        Args:
            common_name: Common name for the certificate
            organization: Organization name
            country: Country code
            key_size: RSA key size in bits

        Returns:
            CSR in PEM format
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=key_size
            )

            # Generate CSR
            subject = cryptography.x509.Name(
                [
                    cryptography.x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                    cryptography.x509.NameAttribute(
                        NameOID.ORGANIZATION_NAME, organization
                    ),
                    cryptography.x509.NameAttribute(NameOID.COUNTRY_NAME, country),
                ]
            )

            csr = (
                cryptography.x509.CertificateSigningRequestBuilder()
                .subject_name(subject)
                .sign(private_key, hashes.SHA256())
            )

            # Save private key
            key_path = Path(self.config.security.tls.key_file or "")
            key_path.parent.mkdir(parents=True, exist_ok=True)

            with open(key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )

            logger.info("csr_generated", common_name=common_name)
            return csr.public_bytes(serialization.Encoding.PEM).decode()
        except Exception as e:
            logger.error("csr_generation_failed", error=str(e))
            raise

    def generate_jwt_token(
        self, user_id: str, expires_in: int = 3600, **kwargs: Any
    ) -> str:
        """
        Generate a JWT token.

        Args:
            user_id: User ID
            expires_in: Token expiration time in seconds
            **kwargs: Additional claims to include in the token

        Returns:
            JWT token
        """
        try:
            if not self.config.security.auth.enabled:
                raise ValueError("JWT authentication is not enabled")

            payload = {
                "sub": user_id,
                "user_id": user_id,
                "exp": datetime.now(UTC) + timedelta(seconds=expires_in),
                **kwargs,
            }

            token = jwt.encode(
                payload,
                self.jwt_secret,
                algorithm=self.config.security.auth.jwt_algorithm,
            )
            logger.info("jwt_token_generated", user_id=user_id)
            return token
        except Exception as e:
            logger.error("jwt_token_generation_failed", error=str(e))
            raise

    def verify_jwt_token(self, token: str) -> dict[str, Any]:
        """
        Verify a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload
        """
        try:
            if not self.config.security.auth.enabled:
                raise ValueError("JWT authentication is not enabled")

            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.config.security.auth.jwt_algorithm],
            )
            logger.info(
                "jwt_token_verified",
                user_id=payload.get("sub") or payload.get("user_id"),
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.error("jwt_token_expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error("jwt_token_invalid", error=str(e))
            raise

    def encrypt_data(self, data: str) -> str:
        """
        Encrypt data using Fernet symmetric encryption.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error("data_encryption_failed", error=str(e))
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data using Fernet symmetric encryption.

        Args:
            encrypted_data: Encrypted data to decrypt

        Returns:
            Decrypted data
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error("data_decryption_failed", error=str(e))
            raise

    def generate_password_hash(
        self, password: str, salt: bytes | None = None
    ) -> dict[str, bytes]:
        """
        Generate a password hash using PBKDF2.

        Args:
            password: Password to hash
            salt: Optional salt (if not provided, a new one will be generated)

        Returns:
            Dict containing hash and salt
        """
        try:
            if salt is None:
                salt = os.urandom(16)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            salt = base64.urlsafe_b64encode(salt)

            return {"hash": key, "salt": salt}
        except Exception as e:
            logger.error("password_hash_generation_failed", error=str(e))
            raise

    def verify_password(self, password: str, stored_hash: bytes, salt: bytes) -> bool:
        """
        Verify a password against its stored hash.

        Args:
            password: Password to verify
            stored_hash: Stored password hash
            salt: Stored salt

        Returns:
            True if password matches, False otherwise
        """
        try:
            try:
                raw_salt = base64.urlsafe_b64decode(salt)
            except Exception:
                raw_salt = salt
            hash_result = self.generate_password_hash(password, raw_salt)
            return hash_result["hash"] == stored_hash
        except Exception as e:
            logger.error("password_verification_failed", error=str(e))
            return False
