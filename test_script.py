"""
Enhanced standalone test for natural language to SQL conversion using OpenAI
This tests the core functionality without MCP complexity
"""

import asyncio
import json
import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

class PostgreSQLQueryEngine:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'testdb'),
            'user': os.getenv('DB_USER', 'testuser'),
            'password': os.getenv('DB_PASSWORD', 'testpass')
        }
        
        # Initialize OpenAI client
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        print("✅ OpenAI client initialized")

    def get_schema(self, table_name: Optional[str] = None) -> str:
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

    async def convert_nl_to_sql_with_ai(self, query: str) -> Dict[str, Any]:
        """Convert natural language to SQL using OpenAI GPT-4o-mini."""
        try:
            # Get schema for context
            schema = self.get_schema()
            
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
7. Use proper data types and casting when necessary
8. Handle NULL values appropriately

Return your response as a JSON object with these fields:
- "sql": The SQL query (required)
- "explanation": Brief explanation of what the query does (required)
- "assumptions": Any assumptions made (optional)
- "confidence": Your confidence level (1-10) in the query correctness (optional)

Example response:
{{
    "sql": "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC LIMIT 10",
    "explanation": "Shows top 10 highest paid employees in Engineering department",
    "assumptions": "Limited to 10 results to avoid large output",
    "confidence": 9
}}
"""

            user_prompt = f"Convert this natural language query to SQL: {query}"

            print(f"🤖 Asking AI to convert: '{query}'")
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent SQL generation
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 AI Response: {response_text[:200]}...")
            
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
                if isinstance(result, dict) and "sql" in result:
                    return result
            except json.JSONDecodeError:
                print("⚠️  Response not in JSON format, trying to extract SQL...")
                
                # If not JSON, try to extract SQL from the response
                sql_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    return {
                        "sql": sql_match.group(1).strip(),
                        "explanation": "Generated from non-JSON AI response"
                    }
                
                # If no SQL block, look for SELECT statements
                select_match = re.search(r'(SELECT\s+.*?(?:;|$))', response_text, re.DOTALL | re.IGNORECASE)
                if select_match:
                    return {
                        "sql": select_match.group(1).strip().rstrip(';'),
                        "explanation": "Extracted SELECT statement from AI response"
                    }
                
                # Last resort: assume entire response is SQL if it starts with SELECT
                if response_text.strip().upper().startswith('SELECT'):
                    return {
                        "sql": response_text.strip().rstrip(';'),
                        "explanation": "Treated entire response as SQL query"
                    }
            
            return {"error": "Could not parse AI response", "raw_response": response_text}
        
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {str(e)}")
            return {"error": f"AI service error: {str(e)}"}

    def is_safe_query(self, sql: str) -> bool:
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
            r'union.*select.*from',  # UNION injections (basic check)
            r'or.*1\s*=\s*1',  # Always true conditions
            r'and.*1\s*=\s*1',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sql_lower):
                return False
        
        return True

    def execute_sql_query(self, sql: str) -> str:
        """Execute SQL query and return formatted results."""
        try:
            # Safety check
            if not self.is_safe_query(sql):
                return "❌ Query contains potentially unsafe operations. Only SELECT queries are allowed."
            
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
                
                # Add row count
                output.append(f"\n({len(results)} row{'s' if len(results) != 1 else ''})")
                
                return "\n".join(output)
            else:  # INSERT, UPDATE, DELETE (shouldn't reach here due to safety check)
                conn.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
        
        except Exception as e:
            return f"❌ Database error: {str(e)}"
        
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    async def handle_natural_language_query(self, query: str, verbose: bool = False) -> str:
        """Convert natural language to SQL using AI and execute."""
        try:
            # Convert natural language to SQL using AI
            ai_result = await self.convert_nl_to_sql_with_ai(query)
            
            if "error" in ai_result:
                return f"❌ Could not interpret: '{query}'\nError: {ai_result['error']}\n\nTry queries like:\n- How many employees are there?\n- Show employees with salary above 50000\n- What is the average salary by department?"
            
            sql = ai_result.get("sql", "").strip()
            explanation = ai_result.get("explanation", "")
            assumptions = ai_result.get("assumptions", "")
            confidence = ai_result.get("confidence")
            
            if not sql:
                return f"❌ No SQL generated for: '{query}'"
            
            # Execute the generated SQL
            print(f"🔄 Executing SQL: {sql}")
            result = self.execute_sql_query(sql)
            
            # Format response
            output = []
            output.append(f"🔍 Query: {query}")
            output.append(f"📝 Generated SQL: {sql}")
            
            if verbose:
                if explanation:
                    output.append(f"💡 Explanation: {explanation}")
                if assumptions:
                    output.append(f"📋 Assumptions: {assumptions}")
                if confidence:
                    output.append(f"🎯 AI Confidence: {confidence}/10")
            
            output.append(f"📊 Result:")
            output.append(result)
            
            return "\n".join(output)
        
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

async def main():
    """Main function to test natural language queries with AI."""
    
    print("🚀 Enhanced PostgreSQL Natural Language Query Engine with OpenAI")
    print("=" * 80)
    
    try:
        engine = PostgreSQLQueryEngine()
        
        # Test database connection
        if not engine.test_connection():
            print("❌ Cannot proceed without database connection")
            return
        
        # Show schema
        print("\n📋 Database Schema:")
        schema = engine.get_schema()
        print(schema)
        print("=" * 80)
        
        # Test queries with AI
        test_queries = [
            "How many employees are there?",
            "Show me employees with salary above 50000",
            "What is the average salary by department?",
            "Who are the top 5 highest paid employees?",
            "List employees hired in the last year",
            "Show me all departments",
            "Find employees in engineering department",
            "What's the salary range in each department?",
            "Show employees with their department information"
        ]
        
        print("🧪 Running AI-powered test queries...")
        print()
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}:")
            result = await engine.handle_natural_language_query(query, verbose=True)
            print(result)
            print("-" * 80)
            
            # Small delay to be respectful to OpenAI API
            await asyncio.sleep(0.5)
        
        # Interactive mode
        print("\n🤖 Interactive Mode (type 'quit' to exit, 'help' for examples)")
        print("Try natural language queries about your database:")
        
        while True:
            try:
                user_query = input("\n🗣️  Your query: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_query.lower() == 'help':
                    print("\n💡 Example queries you can try:")
                    print("• How many employees work in each department?")
                    print("• Show me employees earning more than the average salary")
                    print("• What are the top 3 departments by headcount?")
                    print("• Find all employees hired after 2022")
                    print("• Who has the highest salary in marketing?")
                    print("• Show salary statistics by department")
                    print("• List employees with their manager information")
                    continue
                
                if not user_query:
                    continue
                
                result = await engine.handle_natural_language_query(user_query, verbose=True)
                print(result)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Make sure to set OPENAI_API_KEY in your .env file")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())