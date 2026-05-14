from mcp.server.fastmcp import FastMCP

from agent_eval.core.config import config
from agent_eval.tools.calc import calculate_power, calculate_transmission, convert_speed

mcp = FastMCP(
    config.MCP_NAME,
    stateless_http=True,
    json_response=True,
    host=config.MCP_HOST,
    port=config.MCP_PORT,
)

mcp.tool()(convert_speed)
mcp.tool()(calculate_power)
mcp.tool()(calculate_transmission)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
