"""HTTP helper that routes outgoing calls through MudraID when configured.

Uses the MudraID ``Agent`` (a drop-in for ``requests``) so calls to the
contacts host API are automatically signed with a short-lived bearer
token. When MUDRAID credentials are absent it falls back to plain
``requests`` so the app keeps working without the SDK.

Note on bootstrap: the installed SDK (v0.1.0) discovers the agent's
platforms via ``POST /api/v1/auth/agents/me/platforms``, an endpoint the
current MudraID backend has removed. We inject the platform mapping
directly from ``MUDRAID_PLATFORM_ID`` so the SDK skips that bootstrap;
token minting and the automatic 401-refresh still run through the SDK's
working code paths.
"""

import logging
import os
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

# Make sure MUDRAID_* from .env are in os.environ before the check
# below, regardless of import order in the app.
load_dotenv()

try:
    from mudraid import Agent
    from mudraid.exceptions import MudraIDError
    MUDRAID_AVAILABLE = True
except ImportError:  # mudraid not installed -> run unsigned
    Agent = None

    class MudraIDError(Exception):
        pass

    MUDRAID_AVAILABLE = False

logger = logging.getLogger("http_client")

_agent = None
_mudraid_checked = False


def _get_agent() -> Agent | None:
    global _agent, _mudraid_checked
    if _mudraid_checked:
        return _agent
    _mudraid_checked = True

    if not MUDRAID_AVAILABLE:
        logger.info("mudraid SDK not installed; using plain requests")
        _agent = None
        return None

    key_id = os.getenv("MUDRAID_API_KEY_ID", "").strip()
    secret = os.getenv("MUDRAID_SECRET", "").strip()
    platform_id = os.getenv("MUDRAID_PLATFORM_ID", "").strip()

    if not key_id or not secret:
        logger.info("MudraID credentials not set; using plain requests")
        _agent = None
        return None

    if not platform_id:
        logger.warning(
            "MudraID credentials are set but MUDRAID_PLATFORM_ID is missing; "
            "using plain requests. Set MUDRAID_PLATFORM_ID to the UUID of your "
            "contacts platform in the MudraID portal to activate signing."
        )
        _agent = None
        return None

    host = urlsplit(os.getenv("CONTACTS_API_URL", "")).hostname
    if not host:
        raise MudraIDError("CONTACTS_API_URL has no host; cannot resolve MudraID platform")

    agent = Agent()
    agent._platforms._map = {host.lower(): platform_id}
    logger.info(
        "MudraID agent active: api_key_id=%s platform=%s host=%s",
        agent.api_key_id,
        platform_id,
        host,
    )
    _agent = agent
    return agent


def request(method: str, url: str, **kwargs):
    """Mirror ``requests.request`` but through MudraID when configured."""
    agent = _get_agent()
    if agent is not None:
        return getattr(agent, method.lower())(url, **kwargs)
    return requests.request(method, url, **kwargs)


__all__ = ["request", "MudraIDError", "requests"]
