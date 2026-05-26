from zai import ZhipuAiClient
from dotenv import load_dotenv
from pydantic import BaseModel
from enum import Enum

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
    @property
    def output(self) -> str:
        return f"提示词:{self.prompt_tokens} 回复内容:{self.completion_tokens} 缓存:{self.cache_tokens} 共计:{self.total_tokens}"

class ContentModel(BaseModel):
    content: str
    usage: Usage

class MessageModel(BaseModel):
    role: ROLE
    content: str

def fetch(kwargs:dict) -> ContentModel:
    result = client.chat.completions.create(**kwargs)
    content = result.choices[0].message.content
    prompt_tokens = result.usage.prompt_tokens
    completion_tokens = result.usage.completion_tokens
    cache_tokens = result.usage.prompt_tokens_details.cached_tokens
    total_tokens = result.usage.total_tokens
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_tokens=cache_tokens,
        total_tokens=total_tokens
    )
    return ContentModel(content=content, usage=usage)