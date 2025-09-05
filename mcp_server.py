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
from decimal import Decimal 

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

# Custom JSON Encoder for Decimal and other non-serializable types
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  # or str(obj) if exact precision required
        return super().default(obj)

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
                    # Ensure we use the custom encoder for serialization
                    json_result = json.dumps(result, indent=2, cls=EnhancedJSONEncoder)
                    return [types.TextContent(type="text", text=json_result)]
                
                elif name == "execute_sql":
                    sql = arguments.get("sql", "")
                    result = await self._execute_sql_query(sql)
                    # Ensure we use the custom encoder for serialization
                    json_result = json.dumps(result, indent=2, cls=EnhancedJSONEncoder)
                    return [types.TextContent(type="text", text=json_result)]
                
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

    async def _handle_natural_language_query(self, query: str, explain: bool = False) -> Dict[str, Any]:
        """Convert natural language to SQL using OpenAI and execute."""
        try:
            # Get schema for context
            schema = await self._get_schema()
            
            # Convert natural language to SQL using OpenAI
            sql_result = await self._convert_nl_to_sql_with_ai(query, schema)
            
            if not sql_result.get("sql"):
                return {"error": f"Could not interpret the query: {sql_result.get('error', 'Unknown error')}"}
            
            sql = sql_result["sql"]
            explanation = sql_result.get("explanation", "")
            
            # Validate the generated SQL
            if not self._is_safe_query(sql):
                return {"error": "Generated query contains potentially unsafe operations. Only SELECT queries are allowed."}
            
            # Execute the generated SQL
            execution_result = await self._execute_sql_query(sql)
            
            # Build structured response
            response_obj: Dict[str, Any] = {
                "generated_sql": sql,
                "result": execution_result
            }
            
            if explain and explanation:
                response_obj["explanation"] = explanation
            
            return response_obj
        
        except Exception as e:
            logger.error(f"Error processing natural language query: {str(e)}")
            return {"error": f"Error processing query: {str(e)}"}

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

IMPORTANT: Return ONLY a valid JSON object (no markdown code blocks or extra formatting). The JSON should have these fields:
- "sql": The SQL query (required)
- "explanation": Brief explanation of what the query does (optional)
- "assumptions": Any assumptions made (optional)

Example response:
{{"sql": "SELECT * FROM employees WHERE salary > 50000;", "explanation": "Returns all employees with salary greater than 50000"}}
"""
            user_prompt = f"Convert this natural language query to SQL: {query}"

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Raw AI response: {response_text}")
            
            # Clean up the response - remove markdown code blocks if present
            cleaned_response = response_text
            
            # Remove JSON code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                cleaned_response = json_match.group(1).strip()
                logger.info(f"Extracted JSON from code block: {cleaned_response}")
            
            # Remove generic code blocks
            elif re.match(r'```', response_text):
                code_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', response_text, re.DOTALL)
                if code_match:
                    cleaned_response = code_match.group(1).strip()
                    logger.info(f"Extracted content from code block: {cleaned_response}")
            
            # Try to parse as JSON
            try:
                result = json.loads(cleaned_response)
                if isinstance(result, dict) and "sql" in result:
                    logger.info(f"Successfully parsed AI response: {result}")
                    return result
            except json.JSONDecodeError as je:
                logger.warning(f"JSON decode error: {je}")
                # Fall back to extracting SQL from various formats
                pass
            
            # Fallback: try to extract SQL from various formats
            sql_patterns = [
                r'"sql":\s*"([^"]+)"',  # Extract from JSON-like string
                r'```sql\s*(.*?)\s*```',  # SQL code block
                r'SELECT\s+.*?(?:;|$)',  # Direct SQL statement
            ]
            
            for pattern in sql_patterns:
                sql_match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    extracted_sql = sql_match.group(1) if 'sql' in pattern else sql_match.group(0)
                    logger.info(f"Extracted SQL using pattern {pattern}: {extracted_sql}")
                    return {"sql": extracted_sql.strip()}
            
            # If all else fails, assume the entire response is SQL
            if response_text and ('SELECT' in response_text.upper()):
                logger.info(f"Treating entire response as SQL: {response_text}")
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

    async def _execute_sql_query(self, sql: str) -> Any:
        """Execute SQL query and return structured JSON results."""
        try:
            if not self._is_safe_query(sql):
                return {"error": "Query contains potentially unsafe operations"}
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(sql)
            
            if cursor.description:  # SELECT query
                results = cursor.fetchall()
                return {"rows": results, "row_count": len(results)}
            else:  # Non-SELECT
                conn.commit()
                return {"message": "Query executed successfully", "rows_affected": cursor.rowcount}
        
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
        
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def _is_safe_query(self, sql: str) -> bool:
        """Enhanced SQL injection protection - Fixed to allow legitimate SELECT queries."""
        try:
            sql_clean = sql.strip()
            logger.info(f"Checking safety of SQL: '{sql_clean}'")
            
            # Remove trailing semicolon if present
            if sql_clean.endswith(";"):
                sql_clean = sql_clean[:-1].strip()
                logger.info(f"After removing trailing semicolon: '{sql_clean}'")
            
            # Convert to lowercase for checking
            sql_lower = sql_clean.lower()
            logger.info(f"Lowercase version: '{sql_lower}'")
            
            # Must start with SELECT or WITH (for CTEs)
            select_match = re.match(r'^\s*(select|with)', sql_lower)
            if not select_match:
                logger.warning(f"Query rejected: doesn't start with SELECT or WITH. Starts with: '{sql_lower[:20]}...'")
                return False
            logger.info(f"Query starts correctly with: {select_match.group(1)}")
            
            # Check for multiple statements (semicolons in the middle)
            # Count semicolons - should be 0 after removing trailing one
            semicolon_count = sql_clean.count(';')
            if semicolon_count > 0:
                logger.warning(f"Query rejected: contains {semicolon_count} semicolons (multiple statements): {sql[:50]}...")
                return False
            logger.info("No semicolons found - single statement confirmed")
            
            # Check for dangerous SQL patterns that could modify data
            dangerous_patterns = [
                (r'\bdrop\s+table\b', 'DROP TABLE'),
                (r'\bdrop\s+database\b', 'DROP DATABASE'), 
                (r'\bdrop\s+schema\b', 'DROP SCHEMA'),
                (r'\bdelete\s+from\b', 'DELETE FROM'),
                (r'\btruncate\s+table\b', 'TRUNCATE TABLE'),
                (r'\balter\s+table\b', 'ALTER TABLE'),
                (r'\bcreate\s+table\b', 'CREATE TABLE'),
                (r'\bcreate\s+database\b', 'CREATE DATABASE'),
                (r'\binsert\s+into\b', 'INSERT INTO'),
                (r'\bupdate\s+\w+\s+set\b', 'UPDATE SET'),
                (r'\bgrant\s+', 'GRANT'),
                (r'\brevoke\s+', 'REVOKE'),
                (r'\bexec\s*\(', 'EXEC'),
                (r'\bexecute\s*\(', 'EXECUTE'),
                (r'\bsp_\w+', 'Stored Procedure'),
                (r'\bxp_\w+', 'Extended Procedure'),
                (r'--[^\r\n]*', 'SQL Comment'),  # SQL comments
                (r'/\*.*?\*/', 'Block Comment'),  # Block comments
            ]
            
            for pattern, name in dangerous_patterns:
                match = re.search(pattern, sql_lower, re.IGNORECASE)
                if match:
                    logger.warning(f"Query rejected: matches dangerous pattern '{name}' (pattern: {pattern}). Matched: '{match.group()}'")
                    return False
            logger.info("No dangerous patterns found")
            
            # Additional safety: Check for common SQL injection patterns
            injection_patterns = [
                (r'\bunion\s+select\b.*\bfrom\s+information_schema\b', 'UNION SELECT from information_schema'),
                (r'\bunion\s+select\b.*\bfrom\s+pg_\w+', 'UNION SELECT from PostgreSQL system tables'),
                (r';\s*drop\b', 'Semicolon followed by DROP'),
                (r';\s*delete\b', 'Semicolon followed by DELETE'),
                (r';\s*insert\b', 'Semicolon followed by INSERT'),
                (r';\s*update\b', 'Semicolon followed by UPDATE'),
            ]
            
            for pattern, name in injection_patterns:
                match = re.search(pattern, sql_lower, re.IGNORECASE)
                if match:
                    logger.warning(f"Query rejected: matches injection pattern '{name}' (pattern: {pattern}). Matched: '{match.group()}'")
                    return False
            logger.info("No injection patterns found")
            
            logger.info(f"Query approved as safe: {sql_clean}")
            return True
            
        except Exception as e:
            logger.error(f"Error in _is_safe_query: {str(e)}")
            return False

    async def _get_schema(self, table_name: Optional[str] = None) -> str:
        """Get database schema information."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            if table_name:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
            else:
                cursor.execute("""
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """)
            
            results = cursor.fetchall()
            if not results:
                return "No schema information found"
            
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
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return
        
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