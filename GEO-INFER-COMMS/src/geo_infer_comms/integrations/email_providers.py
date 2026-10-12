"""
Email provider integrations for GEO-INFER-COMMS.

This module provides comprehensive email delivery integrations with
major email service providers including SendGrid, AWS SES, Mailgun,
and others for reliable email notifications and communications.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import StrEnum
from typing import Any, NoReturn, cast
from collections.abc import Callable

import requests

from geo_infer_comms.models.spatial import GeospatialMetadata


class EmailErrorCategory(StrEnum):
    """Stable caller-facing categories for email delivery failures."""

    INVALID_RECIPIENT = "invalid_recipient"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    TRANSPORT = "transport"
    PROVIDER = "provider"


class EmailDeliveryError(RuntimeError):
    """Safe, typed failure returned by an email provider boundary."""

    def __init__(
        self,
        provider: str,
        category: EmailErrorCategory,
        safe_detail: str,
        *,
        cause_type: str | None = None,
    ) -> None:
        self.provider = provider
        self.category = category
        self.safe_detail = safe_detail
        self.cause_type = cause_type
        cause_suffix = f" ({cause_type})" if cause_type else ""
        super().__init__(
            f"{provider} email delivery failed [{category.value}]{cause_suffix}: {safe_detail}"
        )


class EmailProvider(ABC):
    """
    Base class for email provider integrations.

    Provides common functionality for email delivery with geospatial
    context support and comprehensive error handling.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_name = config.get("provider", "generic")
        self.api_key = config.get("api_key")
        self.from_email = config.get("from_email", "noreply@geo-infer.org")
        self.from_name = config.get("from_name", "GEO-INFER Communications")

        # Rate limiting and retry configuration
        self.rate_limit_per_minute = config.get("rate_limit_per_minute", 1000)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 1.0)

        # Performance tracking
        self.emails_sent = 0
        self.emails_failed = 0
        self.last_send_time: float | None = None

        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        geospatial_context: GeospatialMetadata | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """
        Send an email with optional geospatial context.

        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body content
            geospatial_context: Optional geospatial context
            attachments: Optional email attachments

        Returns:
            True if email sent successfully

        Raises:
            EmailDeliveryError: If validation, configuration, dependency,
                transport, or provider execution fails.
        """
        raise RuntimeError("Email provider subclasses must implement send_email")

    def _fail(
        self,
        category: EmailErrorCategory,
        safe_detail: str,
        cause: BaseException | None = None,
    ) -> NoReturn:
        """Record and raise a redacted provider failure."""
        self.emails_failed += 1
        cause_type = type(cause).__name__ if cause is not None else None
        self.logger.error(
            "%s email delivery failed [%s]%s: %s",
            self.provider_name,
            category.value,
            f" ({cause_type})" if cause_type else "",
            safe_detail,
        )
        raise EmailDeliveryError(
            self.provider_name,
            category,
            safe_detail,
            cause_type=cause_type,
        ) from None

    def format_email_with_geospatial_context(
        self,
        subject: str,
        body: str,
        geospatial_context: GeospatialMetadata | None = None,
    ) -> dict[str, str]:
        """
        Format email content with geospatial context information.

        Args:
            subject: Base email subject
            body: Base email body
            geospatial_context: Geospatial context to include

        Returns:
            Formatted email content
        """
        # Enhance subject with geospatial context if available
        enhanced_subject = subject
        if geospatial_context:
            location = geospatial_context.location
            enhanced_subject = f"[Location: {location.latitude:.4f}, {location.longitude:.4f}] {subject}"

        # Enhance body with geospatial details
        enhanced_body = body
        if geospatial_context:
            location_info = f"""
Geospatial Context:
- Location: {geospatial_context.location.latitude:.6f}, {geospatial_context.location.longitude:.6f}
- Accuracy: {geospatial_context.accuracy or "Unknown"} meters
- Source: {geospatial_context.source or "Unknown"}
- Timestamp: {geospatial_context.timestamp.isoformat()}
"""
            enhanced_body += location_info

        return {"subject": enhanced_subject, "body": enhanced_body}

    def validate_email_address(self, email: str) -> bool:
        """Validate email address format."""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def create_mime_message(
        self, to_email: str, subject: str, body: str, html_body: str | None = None
    ) -> MIMEMultipart:
        """Create MIME message for email."""
        msg = MIMEMultipart("alternative")

        # Add text part
        text_part = MIMEText(body, "plain")
        msg.attach(text_part)

        # Add HTML part if provided
        if html_body:
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

        # Set headers
        msg["Subject"] = subject
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = to_email

        return msg

    def get_provider_stats(self) -> dict[str, Any]:
        """Get provider-specific statistics."""
        return {
            "provider": self.provider_name,
            "emails_sent": self.emails_sent,
            "emails_failed": self.emails_failed,
            "success_rate": (
                self.emails_sent / max(self.emails_sent + self.emails_failed, 1) * 100
            ),
        }


class SendGridProvider(EmailProvider):
    """SendGrid email provider integration."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.provider_name = "sendgrid"
        self.api_url = "https://api.sendgrid.com/v3/mail/send"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        geospatial_context: GeospatialMetadata | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send email via SendGrid API."""
        if not self.validate_email_address(to_email):
            self._fail(
                EmailErrorCategory.INVALID_RECIPIENT,
                "recipient address failed syntax validation",
            )
        if not self.api_key:
            self._fail(
                EmailErrorCategory.CONFIGURATION,
                "SendGrid api_key is required",
            )

        try:
            # Format content with geospatial context
            formatted = self.format_email_with_geospatial_context(
                subject, body, geospatial_context
            )

            # Create SendGrid payload
            payload = {
                "personalizations": [
                    {"to": [{"email": to_email}], "subject": formatted["subject"]}
                ],
                "from": {"email": self.from_email, "name": self.from_name},
                "content": [{"type": "text/plain", "value": formatted["body"]}],
            }

            response = await asyncio.to_thread(
                requests.post,
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            self.logger.info(
                "SendGrid email sent to %s: %s", to_email, formatted["subject"]
            )
            self.emails_sent += 1

            return True

        except requests.RequestException as exc:
            self._fail(
                EmailErrorCategory.TRANSPORT,
                "SendGrid request did not complete successfully",
                exc,
            )
        except EmailDeliveryError:
            raise
        except Exception as exc:
            self._fail(
                EmailErrorCategory.PROVIDER,
                "SendGrid client execution failed",
                exc,
            )


class SESProvider(EmailProvider):
    """Amazon SES email provider integration."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.provider_name = "ses"
        self.aws_region = config.get("aws_region", "us-east-1")
        self.aws_access_key = config.get("aws_access_key")
        self.aws_secret_key = config.get("aws_secret_key")

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        geospatial_context: GeospatialMetadata | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send email via Amazon SES."""
        if not self.validate_email_address(to_email):
            self._fail(
                EmailErrorCategory.INVALID_RECIPIENT,
                "recipient address failed syntax validation",
            )
        if not self.aws_access_key or not self.aws_secret_key:
            self._fail(
                EmailErrorCategory.CONFIGURATION,
                "AWS credentials are required for SES",
            )

        try:
            # Format content with geospatial context
            formatted = self.format_email_with_geospatial_context(
                subject, body, geospatial_context
            )

            try:
                import boto3
            except ImportError as exc:
                self._fail(
                    EmailErrorCategory.DEPENDENCY,
                    "boto3 is required for SES delivery",
                    exc,
                )
            client = boto3.client(
                "ses",
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
            )
            await asyncio.to_thread(
                client.send_email,
                Source=self.from_email,
                Destination={"ToAddresses": [to_email]},
                Message={
                    "Subject": {"Data": formatted["subject"]},
                    "Body": {"Text": {"Data": formatted["body"]}},
                },
            )
            self.logger.info("SES email sent to %s: %s", to_email, formatted["subject"])
            self.emails_sent += 1

            return True

        except EmailDeliveryError:
            raise
        except Exception as exc:
            self._fail(
                EmailErrorCategory.PROVIDER,
                "SES client execution failed",
                exc,
            )


class MailgunProvider(EmailProvider):
    """Mailgun email provider integration."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.provider_name = "mailgun"
        self.api_url = f"https://api.mailgun.net/v3/{config.get('domain', 'geo-infer.org')}/messages"
        self.api_username = "api"

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        geospatial_context: GeospatialMetadata | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send email via Mailgun API."""
        if not self.validate_email_address(to_email):
            self._fail(
                EmailErrorCategory.INVALID_RECIPIENT,
                "recipient address failed syntax validation",
            )
        if not self.api_key:
            self._fail(
                EmailErrorCategory.CONFIGURATION,
                "Mailgun api_key is required",
            )

        try:
            # Format content with geospatial context
            formatted = self.format_email_with_geospatial_context(
                subject, body, geospatial_context
            )

            response = await asyncio.to_thread(
                requests.post,
                self.api_url,
                auth=(self.api_username, self.api_key),
                data={
                    "from": f"{self.from_name} <{self.from_email}>",
                    "to": [to_email],
                    "subject": formatted["subject"],
                    "text": formatted["body"],
                },
                timeout=30,
            )
            response.raise_for_status()
            self.logger.info(
                "Mailgun email sent to %s: %s", to_email, formatted["subject"]
            )
            self.emails_sent += 1

            return True

        except requests.RequestException as exc:
            self._fail(
                EmailErrorCategory.TRANSPORT,
                "Mailgun request did not complete successfully",
                exc,
            )
        except EmailDeliveryError:
            raise
        except Exception as exc:
            self._fail(
                EmailErrorCategory.PROVIDER,
                "Mailgun client execution failed",
                exc,
            )


class EmailProviderFactory:
    """Factory for creating email provider instances."""

    @staticmethod
    def create_provider(provider_type: str, config: dict[str, Any]) -> EmailProvider:
        """
        Create an email provider instance.

        Args:
            provider_type: Type of email provider ("sendgrid", "ses", "mailgun")
            config: Provider configuration

        Returns:
            Email provider instance

        Raises:
            ValueError: If provider type is not supported
        """
        providers = {
            "sendgrid": SendGridProvider,
            "ses": SESProvider,
            "mailgun": MailgunProvider,
        }

        provider_class = providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unsupported email provider: {provider_type}")

        return cast(Callable[[dict[str, Any]], EmailProvider], provider_class)(config)

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available email provider types."""
        return ["sendgrid", "ses", "mailgun"]
