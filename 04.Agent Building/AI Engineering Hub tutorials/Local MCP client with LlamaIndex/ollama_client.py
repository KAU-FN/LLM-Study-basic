from httpx import SyncByteStream
import nest_asyncio

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# 중첩된 event loop을 처리하기 위함.
nest_asyncio.apply()

# 1. Local LLM 세팅
# Ollama를 미리 설치해두고, llama3.2를 받아둘 것.
# llama3.2에서 사용한 setting을 그대로 llama_index setting에 적용.
llm = Ollama(model='llama3.2', request_timeout=120.0)
Settings.llm = llm


# # 2. 기본적인 MCP client 초기화
# # MCP client가 로컬 MCP server의 SSE endpoint를 가르키도록 설정.
# # 그리고, 가용한 tool을 show.
# mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
# mcp_tools = McpToolSpec(client = mcp_client)

################# server에 어떤 tool이 있을지 간단하게 확인 #################
# tools = mcp_tools.to_tool_list()
# for tool in tools:
#     print(tool.metadata.name, tool.metadata.description)
################################################################
"""출력해보면 server.py에서 작성한 tool의 description이 아래처럼 출력됨.

add_data 
    SQL INSERT 쿼리를 이용, 앞서 생성한 people table에 새 data를 추가.

    Args:
        query (str): 아래 형식의 SQL INSERT 쿼리:
            INSERT INTO people (name, age, profession)
            VALUES ('John Doe', 30, 'Engineer')
    
    Schema:
        - name: Text field (required)
        - age: Integer field (required)
        - profession: Text field (required)
        Note: 'id' field is auto-generated
    
    Returns:
        bool: 데이터가 올바르게 생성되었다면 True, 그외엔 False.
    
    Example:
        >>> query = '''
        ... INSERT INTO people (name, age, profession)
        ... VALUES ('Alice Smith', 25, 'Developer')
        ... '''
        >>> add_data(query)
        True
    
read_data 
    SQL SELECT 쿼리를 이용해서 people table에 있는 데이터를 read.

    Args:
        query (str, optional): SQL SELECT query. Defaults to "SELECT * FROM people".
            Examples:
            - "SELECT * FROM people"
            - "SELECT name, age FROM people WHERE age > 25"
            - "SELECT * FROM people ORDER BY age DESC"
    
    Returns:
        list: List of tuples containing the query results.
              For default query, tuple format is (id, name, age, profession)
    
    Example:
        >>> # Read all records
        >>> read_data()
        [(1, 'John Doe', 30, 'Engineer'), (2, 'Alice Smith', 25, 'Developer')]
        
        >>> # Read with custom query
        >>> read_data("SELECT name, profession FROM people WHERE age < 30")
        [('Alice Smith', 'Developer')]
"""

# 3. System prompt 설정
# LLM이 tool을 호출하는 방법과, tool을 언제 호출해야 할지 결정해야 할 때 도움을 줌.
SYSTEM_PROMPT = """\
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with Our Database.
"""

# 4. Helper function: get_agent()
# MCP tool 목록과 선택한 LLM과 연결되는 FunctionAgent를 생성.
async def get_agent(tools: McpToolSpec):
    # server에 있는 tool(함수들) 불러오고
    tools = await tools.to_tool_list_async()

    # agent 생성
    agent = FunctionAgent(
        name = 'Agent',
        description = 'An agent that can work with our DB software.',
        tools = tools,  # server에 있는 tool들을 사용
        llm = llm,      # 위에서 생성한 Ollama llm(llama3.2) 사용
        system_prompt = SYSTEM_PROMPT
    )

    return agent

# 5. Helper function: handle_user_message()
# 중간중간의 tool 호출을 보여주고, 최종 응답을 return
async def handle_user_message(
        message_content: str,
        agent: FunctionAgent,
        agent_context: Context,
        verbose: bool = False
):
    handler = agent.run(user_msg=message_content, ctx=agent_context)

    # 받은 event를 stream.
    async for event in handler.stream_events():
        # tool 호출이면 어떤 tool을 호출하는지 보여주고,
        if verbose and type(event) == ToolCall:
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        
        # tool 호출이 끝났다면 그 결과를 보여주기.
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} returned {event.tool_output}")
    
    response = await handler

    return str(response)

async def main():
    # 6. MCP client 초기화 및 Agent build
    # MCP client가 로컬 MCP server의 SSE endpoint를 가르키도록 설정.
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tools = McpToolSpec(client = mcp_client)

    # agent 호출하고
    agent = await get_agent(mcp_tools)

    # agent context 생성
    agent_context = Context(agent)

    # 그리고 이제 실행.
    while True:
        user_input = input("Enter your message: ")
        if user_input == 'exit':
            break

        print("User: ", user_input)

        response =  await handle_user_message(user_input, agent, agent_context, verbose=True)

        print("Agent: ", response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
