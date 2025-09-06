"""
FastAPI wrapper for MCP client to expose natural language to SQL
and other MCP tools as REST endpoints.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import traceback  # ✅ Added for detailed error logs

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

# FastAPI app
app = FastAPI(
    title="MCP FastAPI Integration",
    description="Expose MCP PostgreSQL AI-powered tools over REST APIs",
    version="1.0.0"
)

# Request models
class NLQueryRequest(BaseModel):
    query: str
    explain: Optional[bool] = True

class SQLRequest(BaseModel):
    sql: str

class SchemaRequest(BaseModel):
    table_name: Optional[str] = None


# Utility: Run client session with MCP server
async def run_mcp_tool(tool_name: str, arguments: dict):
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # your server script
        env=None
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()

            # Extract tools properly
            tools = []
            if hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            elif isinstance(tools_response, list):
                tools = tools_response

            if not any(
                hasattr(tool, "name") and tool.name == tool_name for tool in tools
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool '{tool_name}' not found in MCP server"
                )

            response = await session.call_tool(tool_name, arguments)

            # Extract content
            content = []
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, list):
                content = response

            result_texts = []
            for item in content:
                if hasattr(item, 'text'):
                    result_texts.append(item.text)
                elif isinstance(item, tuple) and len(item) >= 2:
                    result_texts.append(item[1])
                else:
                    result_texts.append(str(item))

            return result_texts


# Endpoints

@app.post("/nl-query")
async def natural_language_query(request: NLQueryRequest):
    """
    Convert natural language query to SQL and execute using AI.
    """
    try:
        result = await run_mcp_tool(
            "natural_language_query",
            {"query": request.query, "explain": request.explain}
        )
        return {"results": result}
    except Exception as e:
        print("❌ Error in /nl-query:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-sql")
async def execute_sql(request: SQLRequest):
    """
    Execute raw SQL query.
    """
    try:
        result = await run_mcp_tool("execute_sql", {"sql": request.sql})
        return {"results": result}
    except Exception as e:
        print("❌ Error in /execute-sql:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-schema")
async def get_schema(request: SchemaRequest):
    """
    Retrieve database schema (optionally for a specific table).
    """
    try:
        args = {}
        if request.table_name:
            args["table_name"] = request.table_name
        result = await run_mcp_tool("get_schema", args)
        return {"results": result}
    except Exception as e:
        print("❌ Error in /get-schema:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate-sql")
async def validate_sql(request: SQLRequest):
    """
    Validate SQL query using AI.
    """
    try:
        result = await run_mcp_tool("validate_sql", {"sql": request.sql})
        return {"results": result}
    except Exception as e:
        print("❌ Error in /validate-sql:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
