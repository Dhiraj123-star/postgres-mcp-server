

# 🚀 Enhanced MCP PostgreSQL Server + AI

An **Enhanced Model Context Protocol (MCP) server** that connects **PostgreSQL** with **OpenAI GPT-4o-mini** to enable **natural language to SQL query execution**.

## 🔎 Core Functionality

* 🤖 **AI-Powered SQL Generation**
  Convert natural language queries into safe PostgreSQL `SELECT` statements using GPT-4o-mini.

* 🛡 **Safe Query Execution**
  Only allows **read-only queries** (e.g., `SELECT`, `WITH`), blocking dangerous operations like `DROP`, `DELETE`, `UPDATE`.

* 🗂 **Schema Awareness**
  Automatically retrieves database schema to give AI context for generating accurate SQL queries.

* 📊 **Formatted Results**
  SQL query results are neatly formatted in a table-like output.

* 📝 **SQL Validation via AI**
  Validate raw SQL queries with AI for **syntax correctness**, **schema compatibility**, **performance**, and **security**.

* ⚙️ **Available Tools**

  * `natural_language_query` → Ask database questions in plain English.
  * `execute_sql` → Run raw SQL queries (with safety checks).
  * `get_schema` → Fetch schema details for all or specific tables.
  * `validate_sql` → Validate SQL queries without execution.

* 💬 **Interactive Client**
  The MCP client provides a **command-line interface** to:

  * List available tools.
  * Run natural language queries.
  * Get real-time query results.

---
