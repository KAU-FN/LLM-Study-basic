"""
Complete Python MCP Client Example

This client demonstrates how to:
1. Connect to an MCP server using stdio transport
2. List available tools and resources
3. Call calculator tools
4. Handle responses from the server
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import os
import google.generativeai as genai


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def convert_to_llm_tool(tool):
    # MCP의 inputSchema를 복사해서 가져옵니다.
    # (원본을 직접 수정하지 않기 위해 deep copy를 하거나 필요한 부분만 추출합니다)
    properties = {}
    for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
        # Gemini가 싫어하는 'title' 항목을 제외하고 새로운 딕셔너리 생성
        properties[prop_name] = {
            "type": prop_info.get("type"),
            "description": prop_info.get("description")
        }

    tool_schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": tool.inputSchema.get("required", [])
            }
        }
    }
    return tool_schema


def call_llm(prompt, functions):
    print("🤖 CALLING GEMINI LLM...")
    
    # ==========================================
    # 2. 스키마 변환 (OpenAI 형식 -> Gemini 형식)
    # 이전 단계에서 만든 functions 리스트를 Gemini가 이해하는 형태로 포장합니다.
    # ==========================================
    gemini_function_declarations = []
    for func in functions:
        gemini_function_declarations.append({
            "name": func["function"]["name"],
            "description": func["function"]["description"],
            "parameters": func["function"]["parameters"]
        })
    
    # 도구가 존재할 때만 tools 매개변수에 전달하도록 처리
    gemini_tools = [{"function_declarations": gemini_function_declarations}] if gemini_function_declarations else None

    # ==========================================
    # 3. 모델 설정 및 호출
    # 모델명은 gemini-1.5-pro 또는 gemini-1.5-flash를 추천합니다.
    # ==========================================
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful assistant.",
        tools=gemini_tools
    )

    # 프롬프트(사용자 질문) 전송
    response = model.generate_content(prompt)
    # print(response)
    functions_to_call = []

    # ==========================================
    # 4. 응답 분석 (Tool Call 추출)
    # LLM이 도구를 사용해야겠다고 판단했는지 확인합니다.
    # ==========================================
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            # 응답 파트 중 function_call 속성이 있는 경우
            if part.function_call:
                func_call = part.function_call
                print("🛠️ TOOL CALL: ", func_call.name)
                
                # 주의: OpenAI는 arguments를 JSON '문자열'로 주지만, 
                # Gemini는 이미 파싱된 '딕셔너리(객체)' 형태로 주기 때문에 json.loads가 필요 없습니다.
                args = {key: value for key, value in func_call.args.items()}
                
                functions_to_call.append({ "name": func_call.name, "args": args })

    return functions_to_call


class MCPCalculatorClient:
    def __init__(self):
        # Create server parameters for stdio connection
        self.server_params = StdioServerParameters(
            command="python",  # Executable
            args=["C:/vscode/calculator-server/server.py"],  # Server script
            env=None,  # Optional environment variables
        )

    async def run(self):
        """Main client execution function"""
        print("🚀 Starting MCP Python Client...")

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    print("📡 Connecting to MCP server...")
                    
                    # Initialize the connection
                    await session.initialize()
                    print("✅ Connected to MCP server successfully!")

                    # List available tools
                    await self.list_tools(session)
                    
                    # Test calculator operations
                    await self.test_calculator_operations(session)
                    
                    # List and test resources
                    await self.list_and_test_resources(session)
                    
                    print("\n✨ Client operations completed successfully!")

                    # resources = await session.list_resources()
                    # print("LISTING RESOURCES")
                    # for resource in resources:
                    #     print("Resource: ", resource)

                    # 사용 가능한 도구 목록
                    tools = await session.list_tools()
                    functions = []
                    for tool in tools.tools:
                        # print("Tool: ", tool.name)
                        # print("Tool", tool.inputSchema["properties"])
                        functions.append(convert_to_llm_tool(tool))
                    
                    # prompt = "Add 2 to 20"
                    # prompt = "137이랑 38을 더해줘"
                    prompt = "서울 날씨를 알려줘"
                    # prompt = "농담 하나 던져줘"

                    # 어떤 도구가 모두 필요한지 LLM에 물어보세요, 만약 있다면
                    functions_to_call = call_llm(prompt, functions)

                    # 제안된 함수를 호출하세요
                    for f in functions_to_call:
                        result = await session.call_tool(f["name"], arguments=f["args"])
                        print("TOOLS result: ", result.content)

        except Exception as e:
            print(f"❌ Error running MCP client: {e}")
            raise
    
    async def list_tools(self, session: ClientSession):
        """List all available tools on the server"""
        print("\n📋 Listing available tools:")
        try:
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
        except Exception as e:
            print(f"  Error listing tools: {e}")

    async def test_calculator_operations(self, session: ClientSession):
        """Test various calculator operations"""
        print("\n🧮 Testing Calculator Operations:")

        operations = [
            ("add", {"a": 5, "b": 3}, "Add 5 + 3"),
            ("subtract", {"a": 10, "b": 4}, "Subtract 10 - 4"),
            ("multiply", {"a": 6, "b": 7}, "Multiply 6 × 7"),
            ("divide", {"a": 20, "b": 4}, "Divide 20 ÷ 4"),
            ("help", {}, "Help Information"),
        ]

        for tool_name, arguments, description in operations:
            try:
                result = await session.call_tool(tool_name, arguments=arguments)
                result_text = self.extract_text_result(result)
                
                if tool_name == "help":
                    print(f"\n📖 {description}:")
                    print(result_text)
                else:
                    print(f"{description} = {result_text}")
                    
            except Exception as e:
                print(f"  Error calling {tool_name}: {e}")

    async def list_and_test_resources(self, session: ClientSession):
        """List and test reading resources"""
        print("\n📄 Listing available resources:")
        try:
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  - {resource.name}: {resource.description}")
                print(f"    URI: {resource.uri}")

            # Test reading a resource if available
            if resources.resources:
                first_resource = resources.resources[0]
                print(f"\n📖 Reading resource: {first_resource.name}")
                try:
                    content = await session.read_resource(first_resource.uri)
                    print(f"Resource content: {content}")
                except Exception as e:
                    print(f"  Error reading resource: {e}")
            else:
                print("  No resources available")
                
        except Exception as e:
            print(f"  Error listing resources: {e}")

    def extract_text_result(self, result) -> str:
        """
        Extract text content from a tool result object.

        This method attempts to extract the text content from the `content` attribute
        of the result object. If no text content is found, it falls back to converting
        the result to a string. If an error occurs during extraction, it returns "No result".

        Args:
            result: The result object returned by a tool, which may contain a `content` attribute
                    with text or other types of data.

        Returns:
            A string representing the extracted text content, or a fallback string if no text is found.
        """
        try:
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text') and content_item.text:
                        return content_item.text
                    elif hasattr(content_item, 'type') and content_item.type == "text":
                        return getattr(content_item, 'text', str(content_item))
            
            # Fallback: try to convert to string
            return str(result)
        except Exception:
            return "No result"


async def main():
    """Entry point for the client"""
    client = MCPCalculatorClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())