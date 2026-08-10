"""HTTP helper that routes outgoing calls through MudraID when configured.

Uses the MudraID ``Agent`` (a drop-in for ``requests``) so calls to the
contacts host API are automatically signed with a short-lived bearer
token. When MUDRAID credentials are absent it falls back to plain
``requests`` so the app keeps working without the SDK.

Platform routing is the SDK's native bootstrap: the first call asks
MudraID for the platforms this agent is registered with
(``POST /api/v1/auth/agents/me/platforms``) and matches the request
URL's host to the platform's verified hostname — no MUDRAID_PLATFORM_ID
needed. Grant the platform to the agent in the portal first; otherwise
the call fails with MudraIDPlatformNotRegisteredError.
"""

import logging
import os

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

    if not key_id or not secret:
        logger.info("MudraID credentials not set; using plain requests")
        _agent = None
        return None

    agent = Agent()
    # Refresh platform grants so newly added platforms in the portal
    # are picked up immediately without recreating the Agent object.
    try:
        agent.refresh_platforms()
        logger.info("MudraID agent active (platforms refreshed): api_key_id=%s", agent.api_key_id)
    except Exception as exc:
        logger.warning("MudraID refresh_platforms() failed: %s", exc)
        logger.info("MudraID agent active: api_key_id=%s", agent.api_key_id)
    _agent = agent
    return agent


def reset_agent():
    """Force the MudraID agent to re-initialize on the next request.

    Call this after granting a new platform in the MudraID portal so the
    SDK picks up the updated platform list without a server restart.
    """
    global _agent, _mudraid_checked
    _agent = None
    _mudraid_checked = False
    logger.info("MudraID agent reset; will re-initialize on next request")


def request(method: str, url: str, **kwargs):
    """Mirror ``requests.request`` but through MudraID when configured."""
    agent = _get_agent()
    if agent is not None:
        return getattr(agent, method.lower())(url, **kwargs)
    return requests.request(method, url, **kwargs)


__all__ = ["request", "reset_agent", "MudraIDError", "requests"]
