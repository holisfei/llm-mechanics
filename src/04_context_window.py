from zai import ZhipuAiClient
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel

load_dotenv()
client = ZhipuAiClient()

class ROLE(str, Enum):
    user = "user"
    assistant = "assistant"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_tokens: int = 0
    total_tokens: int = 0


class ContentModel(BaseModel):
    content: str
    usage: Usage

class MessageModel(BaseModel):
    role: ROLE
    content: str

messages: list[MessageModel] = []

def fetch(kwargs:dict) -> ContentModel:
    result = client.chat.completions.create(**kwargs)
    content = result.choices[0].message.content
    prompt_tokens = result.usage.prompt_tokens
    completion_tokens = result.usage.completion_tokens
    cache_tokens = result.usage.prompt_tokens_details.cached_tokens
    total_tokens = result.usage.total_tokens
    return ContentModel(content=content, usage=Usage(prompt_tokens,completion_tokens,cache_tokens,total_tokens))
 
def chat_with_no_memory(input: str):
    params:dict = {
        "model": "glm-4.5-air",
        "messages": [
            {'role': 'user', 'content': input},
        ],
        "stream": False,
    }
    result = fetch(params)
    log = f"提示词:{result.usage.prompt_tokens} 回复内容:{result.usage.completion_tokens} 缓存:{result.usage.cache_tokens} 共计:{result.usage.total_tokens}"
    print(f"{result.content}\n{log}")

def chat_with_memory(input: str):
    messages.append(MessageModel(role=ROLE.user, content=input))

    params:dict = {
        "model": "glm-4.5-air",
        "messages": [m.model_dump() for m in messages],
        "stream": False,
    }
    result = fetch(params)

    messages.append(MessageModel(role=ROLE.assistant, content=result.content))

    print(f"{result.content}\n{result.usage}")


def main():
    # 不带上下文记忆
    chat_with_no_memory("我是小明，记住了")
    chat_with_no_memory("我是谁？")

    # 带上下文记忆
    chat_with_memory("我是小明，记住了")
    chat_with_memory("我是谁？")
    chat_with_memory("我喜欢旅游，我去过马来西亚和印度尼西亚")    
    chat_with_memory("我去过哪里？")

if __name__ == "__main__":
    main()