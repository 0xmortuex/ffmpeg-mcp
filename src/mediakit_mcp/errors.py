"""One error type for every failure an agent can act on.

In mcp >= 2 only a ToolError's message reaches the client; any other
exception is reported as a bare "Error executing tool X". Everything we
raise on purpose therefore derives from ToolError.
"""

from mcp.server.mcpserver.exceptions import ToolError


class MediaError(ToolError):
    """An expected failure with a message written for the agent."""
