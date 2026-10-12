"""LLM proxy policy: model allowlist, size caps, and rate limiting.

A dependency-free policy/guard layer that mirrors the shape of GeoLibre's
``ai-proxy`` Cloudflare Worker (``workers/ai-proxy``): an OpenAI-compatible
LLM endpoint is constrained by a **model allowlist**, a **request-size cap**,
an **output-token cap**, and an optional **per-client rate limit**, so a
GEO-INFER agent or dashboard can call a hosted model without shipping provider
credentials or allowing unbounded requests.

This module contains only the policy logic (pure functions over plain data);
it performs no HTTP calls. It is intended to be used by an endpoint adapter in
``GEO-INFER-AGENT`` or ``GEO-INFER-AI`` when a server-side deployment is
planned. Keeping it dependency-free lets the policy be unit-tested everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Sequence

# Buckets backing LLMProxyPolicy.rate_limit_per_minute, keyed by the policy's
# canonical value tuple so repeated enforcement against the same policy shares
# one window. Explicitly-passed TokenBuckets bypass this registry.
_policy_rate_buckets: dict[tuple, TokenBucket] = {}


class LLMProxyPolicyError(ValueError):
    """Raised when a request is rejected by the LLM proxy policy."""


@dataclass(frozen=True)
class LLMProxyPolicy:
    """Declarative limits for an LLM proxy endpoint.

    Attributes:
        allowed_models: Set of model identifiers the proxy may serve. Requests
            for a model outside this set are rejected.
        max_request_chars: Maximum size (in characters) of the serialised
            request payload.
        max_output_tokens: Maximum number of output tokens allowed.
        rate_limit_per_minute: Optional maximum requests per client (by client
            id) per minute.
    """

    allowed_models: Sequence[str] = ()
    max_request_chars: int = 1_000_000
    max_output_tokens: int = 4096
    rate_limit_per_minute: int | None = None


@dataclass
class TokenBucket:
    """A simple leaky-bucket rate limiter keyed by client id.

    ``limit`` requests are allowed per ``window_seconds``; exceeding the window
    resets the count. This is a deliberately simple in-memory guard — for a
    multi-instance deployment use a shared store (e.g. Redis) instead.
    """

    limit: int
    window_seconds: float = 60.0
    _counter: dict[str, int] = field(default_factory=dict)
    _window_start: float = field(default_factory=time.monotonic)

    def allow(self, client_id: str) -> bool:
        """Return whether ``client_id`` may issue another request now."""
        now = time.monotonic()
        if now - self._window_start >= self.window_seconds:
            self._window_start = now
            self._counter.clear()
        count = self._counter.get(client_id, 0)
        if count >= self.limit:
            return False
        self._counter[client_id] = count + 1
        return True


def check_allowed_model(policy: LLMProxyPolicy, model: str) -> None:
    """Raise :class:`LLMProxyPolicyError` if ``model`` is not allowed."""
    if policy.allowed_models and model not in policy.allowed_models:
        raise LLMProxyPolicyError(
            f"model {model!r} is not in the proxy allowlist: "
            f"{sorted(policy.allowed_models)}"
        )


def check_request_size(policy: LLMProxyPolicy, payload_chars: int) -> None:
    """Raise if the request payload exceeds ``max_request_chars``."""
    if payload_chars > policy.max_request_chars:
        raise LLMProxyPolicyError(
            f"request payload of {payload_chars} chars exceeds the "
            f"{policy.max_request_chars} char cap"
        )


def check_output_tokens(policy: LLMProxyPolicy, requested_tokens: int) -> None:
    """Raise if the requested output token budget exceeds the cap."""
    if requested_tokens > policy.max_output_tokens:
        raise LLMProxyPolicyError(
            f"requested {requested_tokens} output tokens exceeds the "
            f"{policy.max_output_tokens} token cap"
        )


def enforce_llm_proxy_policy(
    policy: LLMProxyPolicy,
    *,
    model: str,
    request_payload: str | None = None,
    payload_chars: int | None = None,
    requested_output_tokens: int = 0,
    client_id: str | None = None,
    rate_limiter: TokenBucket | None = None,
) -> None:
    """Apply the full policy guard to one request.

    Checks (in order): model allowlist, request size, output-token cap, and
    the per-client rate limit. Rate limiting consults the supplied
    ``rate_limiter``; when none is given and the policy declares
    ``rate_limit_per_minute``, a shared per-policy bucket is created and
    reused across calls. No rate limiting applies without a ``client_id``.
    Args:
        policy: The active policy.
        model: Requested model identifier.
        request_payload: Optional request payload string used to measure size.
        payload_chars: Alternative to ``request_payload`` — an explicit char
            count (used when the payload is already serialised).
        requested_output_tokens: Requested output token budget.
        client_id: Client identifier for rate limiting.
        rate_limiter: Optional rate limiter to consult; defaults to a shared
            per-policy bucket built from ``policy.rate_limit_per_minute``.

    Raises:
        LLMProxyPolicyError: On any policy violation.
    """
    check_allowed_model(policy, model)

    if payload_chars is None:
        payload_chars = len(request_payload) if request_payload is not None else 0
    check_request_size(policy, payload_chars)

    if requested_output_tokens:
        check_output_tokens(policy, requested_output_tokens)

    if client_id is None:
        return

    if rate_limiter is None:
        if policy.rate_limit_per_minute is None:
            return
        key = (
            tuple(policy.allowed_models),
            policy.max_request_chars,
            policy.max_output_tokens,
            policy.rate_limit_per_minute,
        )
        bucket = _policy_rate_buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(limit=policy.rate_limit_per_minute)
            _policy_rate_buckets[key] = bucket
        rate_limiter = bucket

    if not rate_limiter.allow(client_id):
        raise LLMProxyPolicyError(f"rate limit exceeded for client {client_id!r}")


__all__ = [
    "LLMProxyPolicyError",
    "LLMProxyPolicy",
    "TokenBucket",
    "check_allowed_model",
    "check_request_size",
    "check_output_tokens",
    "enforce_llm_proxy_policy",
]
