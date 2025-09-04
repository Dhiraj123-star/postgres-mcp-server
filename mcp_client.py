import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

async def main():
    # Define how to launch the MCP server over stdio
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # your server script
        env=None  # optional: env variables
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the MCP session
            await session.initialize()

            # List tools exposed by the server
            tools_response = await session.list_tools()
            print("Raw list_tools response:", tools_response)  # Debug output

            # Handle the ListToolsResult object
            tools = []
            if hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            elif isinstance(tools_response, list):
                tools = tools_response
            else:
                print(f"Error: Unexpected response format from list_tools: {type(tools_response)}")
                return

            # Print available tools
            print("Available tools:")
            if not tools:
                print("No tools found in response")
                return

            for tool in tools:
                try:
                    print(f"- {tool.name}: {tool.description}")
                except AttributeError as e:
                    print(f"Error: Invalid tool object format: {e}")
                    print(f"Tool object: {tool}")
                    continue

            # Check if natural_language_query tool is available
            if not any(tool.name == "natural_language_query" for tool in tools if hasattr(tool, 'name')):
                print("\nError: natural_language_query tool not found")
                return

            # Interactive loop for user queries
            print("\nEnter a natural language query (type 'quit' to exit):")
            while True:
                nl_query = input("> ").strip()
                if nl_query.lower() == "quit":
                    print("Exiting...")
                    break

                if not nl_query:
                    print("Please enter a valid query.")
                    continue

                # Call the natural_language_query tool
                response = await session.call_tool(
                    "natural_language_query",
                    {"query": nl_query, "explain": True}
                )
                print("\nRaw Natural Language Query Response:", response)  # Debug output
                print("\nNatural Language Query Response:")
                
                # Handle the CallToolResult object
                content = []
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, list):
                    content = response
                else:
                    print(f"Error: Unexpected response format from call_tool: {type(response)}")
                    continue

                # Process the content list
                if not content:
                    print("No content found in response")
                    continue

                for item in content:
                    try:
                        # Check if item is a TextContent object with a text attribute
                        if hasattr(item, 'text'):
                            print(item.text)
                        # Handle case where item is a tuple (e.g., (type, text))
                        elif isinstance(item, tuple) and len(item) >= 2:
                            print(item[1])  # Assume the second element is the text
                        else:
                            print(f"Unexpected response item format: {item}")
                    except Exception as e:
                        print(f"Error processing response item: {e}")
                        print(f"Item: {item}")

if __name__ == "__main__":
    asyncio.run(main())