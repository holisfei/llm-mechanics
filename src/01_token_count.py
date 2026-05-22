import os
import asyncio
import tiktoken
import httpx
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass # class是类级别的共享属性，不是每个实例独立的，使用 @dataclass
class TokenCount:
    text: str = ""
    count: int = 0
    source: str = ""

async def api_token(input: str, client: httpx.AsyncClient) -> TokenCount:
    params = {
        "model":"glm-4.5-air",
        # "temperature": 1,
        "max_tokens": 1,
        "stream_options": {"include_usage": True},
        "stream": False,
        "messages":[
           {
                "role": "user",
                "content": input
           }
        ]
    }
    
    response = await client.post(
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        json=params,
        follow_redirects=True
    )
    response.raise_for_status()   # 加一行:网络/鉴权出错时能立刻看到,而不是后面 KeyError
    res_json = response.json()
    return TokenCount(text=input, count=res_json["usage"]["prompt_tokens"], source="api")

async def tiktoken_calculate(input: str) -> TokenCount:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(input))
    return TokenCount(text=input, count=tokens, source="tiktoken")

async def tokens_counts(inputs: list[str]) -> list[TokenCount]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ZAI_API_KEY')}"
    }
    timeout = httpx.Timeout(connect=30, read=60, write=60, pool=60)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:

        tasks_tik = [asyncio.create_task(tiktoken_calculate(t)) for t in inputs]
        tasks_api = [asyncio.create_task(api_token(input=t, client=client)) for t in inputs]

        tokens: list[TokenCount] = []
        for task in asyncio.as_completed(tasks_tik+tasks_api):
            result = await task
            tokens.append(result)
    
        return tokens

         
async def main():
    samples = [
        "Hello, how are you today?",
        "你好,你今天过得怎么样?",
        "func add(a: Int, b: Int) -> Int { return a + b }",
        '{"name": "Claude", "age": 2}',
    ]

    tokens: list[TokenCount] = await tokens_counts(inputs=samples)

    for prompt in samples:
        datas = [tc for tc in tokens if tc.text == prompt]
        count_str: str = f"{prompt} 👉🏻 "
        for t in datas:
            count_str = count_str + f"token数:{t.count}(统计来源:{t.source}) "
        print(f"{count_str}")

        # 结果: 
        # Hello, how are you today? 👉🏻 token数:7(统计来源:tiktoken) token数:12(统计来源:api) 
        # 你好,你今天过得怎么样? 👉🏻 token数:13(统计来源:tiktoken) token数:12(统计来源:api) 
        # func add(a: Int, b: Int) -> Int { return a + b } 👉🏻 token数:18(统计来源:tiktoken) token数:23(统计来源:api) 
        # {"name": "Claude", "age": 2} 👉🏻 token数:13(统计来源:tiktoken) token数:18(统计来源:api) 

if __name__ == "__main__":
    asyncio.run(main())