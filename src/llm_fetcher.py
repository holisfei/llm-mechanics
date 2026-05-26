from zai import ZhipuAiClient
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from enum import Enum
import numpy as np

load_dotenv()
client = ZhipuAiClient()

class ROLE(str, Enum):
    user = "user"
    assistant = "assistant"

class Usage(BaseModel):
    # ✅ 允许从 SDK 的 CompletionUsage 对象中提取属性
    model_config = ConfigDict(from_attributes=True) 
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

class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    # ✅ 允许从 SDK 的 Embedding 对象中提取属性
    model_config = ConfigDict(from_attributes=True) 

class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]
    usage: Usage
    # ✅ 允许从已有对象的属性中提取数据
    model_config = ConfigDict(from_attributes=True) 

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

def embeddings(inputs:list[str], dimensions: int = 2048) -> list[EmbeddingData]:
    """对一组文本向量化"""
    response = client.embeddings.create(
        model="embedding-3",
        input=inputs,
        dimensions=dimensions
    )
    model = EmbeddingResponse.model_validate(response)
    return model.data

def calculate_similarity(a_embedding: list[float], b_embedding: list[float]) -> float:
    """计算两个向量的相似度"""
    # 0. 生成向量
    vec_a = np.array(a_embedding)
    vec_b = np.array(b_embedding)

    # 1. 计算 点积
    dot_product = np.dot(vec_a, vec_b)

    # 3. 计算 模长
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # 4. 相似度
    similarity = dot_product / (norm_a * norm_b)

    return similarity