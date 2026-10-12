from __future__ import annotations

from starlette.websockets import WebSocketDisconnect

try:
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    ClientDisconnected = ()  # type: ignore[misc, assignment]


def is_client_disconnect(exc: BaseException) -> bool:
    """True when the peer closed the WebSocket (not a server bug)."""
    if isinstance(exc, WebSocketDisconnect):
        return True
    if ClientDisconnected and isinstance(exc, ClientDisconnected):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        if "disconnect" in text or "connection is closed" in text:
            return True
    return False
