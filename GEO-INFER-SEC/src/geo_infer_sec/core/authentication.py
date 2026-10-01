"""
Authentication system for GEO-INFER-SEC.

This module provides comprehensive authentication capabilities including
OAuth 2.0, JWT token management, and multi-factor authentication.
"""

import logging
import hashlib
import hmac
import secrets
import struct
from typing import Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)

# RFC 6238 TOTP parameters.
_TOTP_STEP_SECONDS = 30
_TOTP_DIGITS = 6
_TOTP_ALLOWED_SKEW = 1  # accept codes from the previous/current/next step


def generate_totp(secret: str, at_time: float | None = None) -> str:
    """Generate an RFC 6238 TOTP code for a base32-encoded secret.

    Args:
        secret: Base32-encoded shared secret (as provisioned via
            ``AuthenticationManager.enable_mfa``).
        at_time: Unix timestamp to generate the code for (defaults to now).

    Returns:
        The zero-padded 6-digit TOTP code.
    """
    key = base64.b32decode(secret, casefold=True)
    counter = int(at_time if at_time is not None else datetime.now(UTC).timestamp())
    counter //= _TOTP_STEP_SECONDS
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (
        ((digest[offset] & 0x7F) << 24)
        | (digest[offset + 1] << 16)
        | (digest[offset + 2] << 8)
        | digest[offset + 3]
    ) % (10**_TOTP_DIGITS)
    return str(code).zfill(_TOTP_DIGITS)


def verify_totp(secret: str, code: str, at_time: float | None = None) -> bool:
    """Verify an RFC 6238 TOTP code against a base32-encoded secret.

    Accepts codes from the previous, current, or next time step to tolerate
    clock skew. Comparison is constant-time.
    """
    try:
        reference = at_time if at_time is not None else datetime.now(UTC).timestamp()
        for skew in range(-_TOTP_ALLOWED_SKEW, _TOTP_ALLOWED_SKEW + 1):
            expected = generate_totp(secret, reference + skew * _TOTP_STEP_SECONDS)
            if hmac.compare_digest(expected, code.strip()):
                return True
        return False
    except Exception as e:
        logger.error(f"TOTP verification error: {e}")
        return False


logger = logging.getLogger(__name__)


@dataclass
class UserCredentials:
    """User credentials for authentication."""

    user_id: str
    username: str
    password_hash: str
    email: str | None = None
    enabled: bool = True
    locked: bool = False
    failed_attempts: int = 0
    last_login: datetime | None = None
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    password_salt: str | None = None


@dataclass
class TokenInfo:
    """JWT token information."""

    token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: str | None = None
    scope: list[str] | None = None


class AuthenticationManager:
    """
    Authentication manager for user authentication and token management.

    Provides OAuth 2.0 compatible authentication, JWT token generation,
    password hashing, and multi-factor authentication support.
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_expiration_hours: int = 24,
        refresh_token_expiration_days: int = 30,
        password_min_length: int = 8,
    ) -> None:
        """
        Initialize the authentication manager.

        Args:
            secret_key: Secret key for JWT token signing
            algorithm: JWT signing algorithm
            token_expiration_hours: Access token expiration in hours
            refresh_token_expiration_days: Refresh token expiration in days
            password_min_length: Minimum password length requirement
        """
        self.secret_key = secret_key
        # PyJWT warns for HMAC keys below the RFC 7518 recommendation. Hash
        # caller-provided material into a stable 32-byte signing key while
        # retaining the original value for configuration introspection.
        self._jwt_secret = hashlib.sha256(secret_key.encode("utf-8")).digest()
        self.algorithm = algorithm
        self.token_expiration_hours = token_expiration_hours
        self.refresh_token_expiration_days = refresh_token_expiration_days
        self.password_min_length = password_min_length

        # In-memory user store (in production, use a database)
        self.users: dict[str, UserCredentials] = {}
        self.refresh_tokens: dict[str, dict[str, Any]] = {}

    def hash_password(
        self, password: str, salt: bytes | None = None
    ) -> tuple[str, str]:
        """
        Hash a password using PBKDF2.

        Args:
            password: Plain text password
            salt: Optional salt (if None, generates a new one)

        Returns:
            Tuple of (password_hash, salt) as base64 encoded strings
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )

        password_hash = base64.b64encode(kdf.derive(password.encode())).decode()
        salt_b64 = base64.b64encode(salt).decode()

        return password_hash, salt_b64

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Plain text password to verify
            password_hash: Stored password hash
            salt: Salt used for hashing

        Returns:
            True if password matches, False otherwise
        """
        try:
            salt_bytes = base64.b64decode(salt)
            expected_hash, _ = self.hash_password(password, salt_bytes)
            return expected_hash == password_hash
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def register_user(
        self,
        username: str,
        password: str,
        email: str | None = None,
        user_id: str | None = None,
    ) -> UserCredentials:
        """
        Register a new user.

        Args:
            username: Username
            password: Plain text password
            email: Optional email address
            user_id: Optional user ID (if None, generates one)

        Returns:
            UserCredentials object

        Raises:
            ValueError: If password doesn't meet requirements or user already exists
        """
        if len(password) < self.password_min_length:
            raise ValueError(
                f"Password must be at least {self.password_min_length} characters"
            )

        if username in self.users:
            raise ValueError(f"User {username} already exists")

        user_id = user_id or f"user_{secrets.token_hex(8)}"
        password_hash, salt = self.hash_password(password)

        user = UserCredentials(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            email=email,
            enabled=True,
        )

        # Store the password salt with the user (in production, store in database).
        # This is deliberately separate from ``mfa_secret`` so enabling MFA can
        # never clobber the password salt.
        user.password_salt = salt

        self.users[username] = user
        logger.info(f"Registered user: {username}")

        return user

    def authenticate(
        self, username: str, password: str, mfa_code: str | None = None
    ) -> TokenInfo | None:
        """
        Authenticate a user and generate access token.

        Args:
            username: Username
            password: Plain text password
            mfa_code: Optional multi-factor authentication code

        Returns:
            TokenInfo if authentication successful, None otherwise
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: user {username} not found")
            return None

        user = self.users[username]

        if not user.enabled:
            logger.warning(f"Authentication failed: user {username} is disabled")
            return None

        if user.locked:
            logger.warning(f"Authentication failed: user {username} is locked")
            return None

        # Verify password
        if not user.password_salt:
            logger.error(f"Password salt not found for user {username}")
            return None

        if not self.verify_password(password, user.password_hash, user.password_salt):
            user.failed_attempts += 1
            if user.failed_attempts >= 5:
                user.locked = True
                logger.warning(f"User {username} locked due to failed attempts")
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None

        # Verify MFA if enabled. A missing or invalid code always denies
        # authentication -- the second factor can never be bypassed by omitting it.
        if user.mfa_enabled:
            if not mfa_code:
                logger.warning(
                    f"Authentication failed: MFA code required for {username}"
                )
                return None
            if not user.mfa_secret or not verify_totp(user.mfa_secret, mfa_code):
                logger.warning(
                    f"Authentication failed: invalid MFA code for {username}"
                )
                return None

        # Reset failed attempts on successful authentication
        user.failed_attempts = 0
        now = datetime.now(UTC).replace(tzinfo=None)
        user.last_login = now

        # Generate tokens
        token_info = self.generate_tokens(user.user_id, username)

        logger.info(f"User {username} authenticated successfully")
        return token_info

    def generate_tokens(
        self, user_id: str, username: str, scope: list[str] | None = None
    ) -> TokenInfo:
        """
        Generate access and refresh tokens.

        Args:
            user_id: User identifier
            username: Username
            scope: Optional list of scopes

        Returns:
            TokenInfo with access and refresh tokens
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        # Generate access token
        access_token_payload = {
            "sub": user_id,
            "username": username,
            "iat": now,
            "exp": now + timedelta(hours=self.token_expiration_hours),
            "scope": scope or ["read", "write"],
        }

        access_token = jwt.encode(
            access_token_payload, self._jwt_secret, algorithm=self.algorithm
        )

        # Generate refresh token
        refresh_token = secrets.token_urlsafe(32)
        refresh_token_payload = {
            "sub": user_id,
            "username": username,
            "token_type": "refresh",
            "exp": now + timedelta(days=self.refresh_token_expiration_days),
        }

        self.refresh_tokens[refresh_token] = refresh_token_payload

        return TokenInfo(
            token=access_token,
            expires_in=self.token_expiration_hours * 3600,
            refresh_token=refresh_token,
            scope=scope or ["read", "write"],
        )

    def validate_token(self, token: str) -> dict[str, Any] | None:
        """
        Validate a JWT access token.

        Args:
            token: JWT token string

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self._jwt_secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> TokenInfo | None:
        """
        Generate a new access token from a refresh token.

        Args:
            refresh_token: Refresh token string

        Returns:
            New TokenInfo if refresh token is valid, None otherwise
        """
        if refresh_token not in self.refresh_tokens:
            logger.warning("Refresh token validation failed: token not found")
            return None

        refresh_payload = self.refresh_tokens[refresh_token]

        # Check expiration
        exp = refresh_payload.get("exp")
        now = datetime.now(UTC).replace(tzinfo=None)
        if exp and now > (
            exp if isinstance(exp, datetime) else datetime.fromtimestamp(exp)
        ):
            del self.refresh_tokens[refresh_token]
            logger.warning("Refresh token validation failed: token expired")
            return None

        # Generate new access token
        user_id = refresh_payload.get("sub")
        username = refresh_payload.get("username")

        if not user_id or not username:
            logger.warning("Refresh token validation failed: invalid payload")
            return None

        return self.generate_tokens(user_id, username)

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a refresh token.

        Args:
            token: Refresh token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        if token in self.refresh_tokens:
            del self.refresh_tokens[token]
            logger.info("Refresh token revoked")
            return True
        return False

    def get_user(self, username: str) -> UserCredentials | None:
        """
        Get user credentials by username.

        Args:
            username: Username

        Returns:
            UserCredentials if found, None otherwise
        """
        return self.users.get(username)

    def enable_mfa(self, username: str, secret: str) -> bool:
        """
        Enable multi-factor authentication for a user.

        Args:
            username: Username
            secret: MFA secret

        Returns:
            True if MFA enabled successfully, False otherwise
        """
        if username not in self.users:
            return False

        try:
            base64.b32decode(secret, casefold=True)
        except Exception as e:
            raise ValueError(
                "MFA secret must be a valid base32 string (RFC 6238 shared secret)"
            ) from e

        user = self.users[username]
        user.mfa_enabled = True
        user.mfa_secret = secret

        logger.info(f"MFA enabled for user {username}")
        return True

    def disable_mfa(self, username: str) -> bool:
        """
        Disable multi-factor authentication for a user.

        Args:
            username: Username

        Returns:
            True if MFA disabled successfully, False otherwise
        """
        if username not in self.users:
            return False

        user = self.users[username]
        user.mfa_enabled = False
        user.mfa_secret = None

        logger.info(f"MFA disabled for user {username}")
        return True
