"""
Enhanced MCP PostgreSQL Server with OpenAI GPT-4o-mini integration
Natural Language to SQL Query Interface using AI
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-postgres")

class PostgreSQLMCPServer:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'testdb'),
            'user': os.getenv('DB_USER', 'testuser'),
            'password': os.getenv('DB_PASSWORD', 'testpass')
        }
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        self.server = Server("postgres-mcp-server")
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="natural_language_query",
                    description="Execute natural language queries against PostgreSQL database using AI interpretation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about the database"
                            },
                            "explain": {
                                "type": "boolean",
                                "description": "Whether to explain the generated SQL query",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="execute_sql",
                    description="Execute raw SQL query against PostgreSQL database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to execute"
                            }
                        },
                        "required": ["sql"]
                    }
                ),
                types.Tool(
                    name="get_schema",
                    description="Get database schema information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Optional specific table name to get schema for"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="validate_sql",
                    description="Validate SQL query using AI without executing it",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to validate"
                            }
                        },
                        "required": ["sql"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """Handle tool calls."""
            try:
                if name == "natural_language_query":
                    query = arguments.get("query", "")
                    explain = arguments.get("explain", False)
                    result = await self._handle_natural_language_query(query, explain)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "execute_sql":
                    sql = arguments.get("sql", "")
                    result = await self._execute_sql_query(sql)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "get_schema":
                    table_name = arguments.get("table_name")
                    result = await self._get_schema(table_name)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "validate_sql":
                    sql = arguments.get("sql", "")
                    result = await self._validate_sql_with_ai(sql)
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_natural_language_query(self, query: str, explain: bool = False) -> str:
        """Convert natural language to SQL using OpenAI and execute."""
        try:
            # Get schema for context
            schema = await self._get_schema()
            
            # Convert natural language to SQL using OpenAI
            sql_result = await self._convert_nl_to_sql_with_ai(query, schema)
            
            if not sql_result.get("sql"):
                return f"Could not interpret the query: {sql_result.get('error', 'Unknown error')}"
            
            sql = sql_result["sql"]
            explanation = sql_result.get("explanation", "")
            
            # Validate the generated SQL
            if not self._is_safe_query(sql):
                return "Generated query contains potentially unsafe operations. Only SELECT queries are allowed."
            
            # Execute the generated SQL
            execution_result = await self._execute_sql_query(sql)
            
            # Format response
            response_parts = []
            
            if explain and explanation:
                response_parts.append(f"Explanation: {explanation}")
            
            response_parts.append(f"Generated SQL: {sql}")
            response_parts.append(f"\nResult:\n{execution_result}")
            
            return "\n\n".join(response_parts)
        
        except Exception as e:
            logger.error(f"Error processing natural language query: {str(e)}")
            return f"Error processing query: {str(e)}"

    async def _convert_nl_to_sql_with_ai(self, query: str, schema: str) -> Dict[str, Any]:
        """Convert natural language to SQL using OpenAI GPT-4o-mini."""
        try:
            system_prompt = f"""
You are an expert SQL query generator. Convert natural language questions into PostgreSQL queries.

Database Schema:
{schema}

Rules:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, DROP, or other modifying operations
2. Use proper PostgreSQL syntax
3. Return valid SQL that can be executed directly
4. If the question is ambiguous, make reasonable assumptions
5. Use appropriate JOINs when querying multiple tables
6. Include LIMIT clauses for potentially large result sets when appropriate

Return your response as a JSON object with these fields:
- "sql": The SQL query (required)
- "explanation": Brief explanation of what the query does (optional)
- "assumptions": Any assumptions made (optional)

Example response:
{{
    "sql": "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC LIMIT 10",
    "explanation": "Shows top 10 highest paid employees in Engineering department",
    "assumptions": "Limited to 10 results to avoid large output"
}}
"""

            user_prompt = f"Convert this natural language query to SQL: {query}"

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent SQL generation
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
                if isinstance(result, dict) and "sql" in result:
                    return result
            except json.JSONDecodeError:
                # If not JSON, try to extract SQL from the response
                sql_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    return {"sql": sql_match.group(1).strip()}
                
                # If no SQL block, assume the entire response is SQL
                if response_text and not response_text.startswith('{'):
                    return {"sql": response_text.strip()}
            
            return {"error": "Could not parse AI response", "raw_response": response_text}
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"error": f"AI service error: {str(e)}"}

    async def _validate_sql_with_ai(self, sql: str) -> str:
        """Validate SQL query using AI without executing it."""
        try:
            schema = await self._get_schema()
            
            system_prompt = f"""
You are a SQL validator. Analyze the given SQL query for:
1. Syntax correctness (PostgreSQL)
2. Schema compatibility
3. Potential performance issues
4. Security concerns

Database Schema:
{schema}

Return a detailed analysis including:
- Whether the query is valid
- Any syntax errors
- Performance recommendations
- Security warnings
"""

            user_prompt = f"Validate this SQL query: {sql}"

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error validating SQL with AI: {str(e)}")
            return f"Error validating SQL: {str(e)}"

    async def _execute_sql_query(self, sql: str) -> str:
        """Execute SQL query and return formatted results."""
        try:
            # Basic SQL injection protection
            if not self._is_safe_query(sql):
                return "Error: Query contains potentially unsafe operations"
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(sql)
            
            if cursor.description:  # SELECT query
                results = cursor.fetchall()
                if not results:
                    return "No results found"
                
                # Format results as table
                headers = [desc[0] for desc in cursor.description]
                
                # Create formatted output
                output = []
                output.append(" | ".join(headers))
                output.append("-" * (len(" | ".join(headers))))
                
                for row in results:
                    row_data = []
                    for col in headers:
                        value = row[col]
                        if value is None:
                            row_data.append("NULL")
                        else:
                            row_data.append(str(value))
                    output.append(" | ".join(row_data))
                
                return "\n".join(output)
            else:  # INSERT, UPDATE, DELETE
                conn.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
        
        except Exception as e:
            return f"Database error: {str(e)}"
        
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def _is_safe_query(self, sql: str) -> bool:
        """Enhanced SQL injection protection."""
        dangerous_keywords = [
            'drop', 'delete', 'truncate', 'alter', 'create', 
            'insert', 'update', 'grant', 'revoke', 'exec',
            'execute', 'sp_', 'xp_', '--', '/*', '*/'
        ]
        
        sql_lower = sql.lower().strip()
        
        # Must start with SELECT (with optional whitespace and comments)
        if not re.match(r'^\s*(select|with)', sql_lower):
            return False
        
        # Check for dangerous operations
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        # Additional checks for common injection patterns
        suspicious_patterns = [
            r';.*select',  # Multiple statements
            r'union.*select.*from',  # UNION injections
            r'or.*1\s*=\s*1',  # Always true conditions
            r'and.*1\s*=\s*1',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sql_lower):
                return False
        
        return True

    async def _get_schema(self, table_name: Optional[str] = None) -> str:
        """Get database schema information."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            if table_name:
                # Get specific table schema
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
            else:
                # Get all tables and their columns
                cursor.execute("""
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """)
            
            results = cursor.fetchall()
            
            if not results:
                return "No schema information found"
            
            # Format schema info
            if table_name:
                output = [f"Schema for table '{table_name}':"]
                for row in results:
                    col_name, data_type, nullable, default = row
                    nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                    default_str = f" DEFAULT {default}" if default else ""
                    output.append(f"  {col_name}: {data_type} {nullable_str}{default_str}")
            else:
                current_table = None
                output = ["Database Schema:"]
                
                for row in results:
                    table, column, data_type, nullable = row
                    
                    if table != current_table:
                        current_table = table
                        output.append(f"\nTable: {table}")
                    
                    nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                    output.append(f"  {column}: {data_type} {nullable_str}")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Error getting schema: {str(e)}"
        
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    async def run(self):
        """Run the MCP server."""
        try:
            # Test database connection first
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return
        
        # Test OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable not set")
            return
        
        logger.info("Starting Enhanced PostgreSQL MCP Server with OpenAI integration...")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="postgres-mcp-server-ai",
                    server_version="2.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

async def main():
    """Main entry point."""
    server = PostgreSQLMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()