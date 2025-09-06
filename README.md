
# 🚀 Enhanced MCP PostgreSQL Server + AI + FastAPI

An **Enhanced Model Context Protocol (MCP) server** that connects **PostgreSQL** with **OpenAI GPT-4o-mini**, now extended with **FastAPI endpoints** to enable **natural language to SQL query execution** via both **MCP client** and **REST API**.

---

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
  Validate raw SQL queries with AI for **syntax correctness**, **schema compatibility**, and **performance**.

---

## ⚡ New Enhancement: FastAPI Endpoints

* **Natural Language Query Endpoint** → Convert plain English into SQL and get results.
* **Execute SQL Endpoint** → Run raw SQL queries (restricted to safe operations).
* **Get Schema Endpoint** → Retrieve schema details for all or specific tables.
* **Validate SQL Endpoint** → Validate SQL queries without executing them.

---

## ⚙️ Available Tools (MCP + API)

* `natural_language_query` → Ask database questions in plain English.
* `execute_sql` → Run raw SQL queries (with safety checks).
* `get_schema` → Fetch schema details for all or specific tables.
* `validate_sql` → Validate SQL queries without execution.

---

## 💬 Interactive Client + REST API

* **MCP Client** → Use CLI to run tools interactively.
* **FastAPI** → Access all tools via RESTful endpoints.

---

## 🔄 CI/CD Pipeline Integration

* **GitHub Actions Workflow** → Automated builds and tests on every push.
* **Docker Image Build & Push** → Publishes latest images to **DockerHub (`dhiraj918106/...`)**.
* **Deployment Ready** → Simple, production-style pipeline to keep images updated.

---


