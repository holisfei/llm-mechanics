# RAG 检索 - 搜索语意相似性片段

- 为什么 RAG 要存在？因为模型没记忆,记忆数据得自己造，然后喂给模型
- 靠prompt引导型别瞎编不可靠，真正的解法是 RAG 把真实资料喂给它
- RAG 从外部知识库检索相关文档片段,作为上下文喂给模型

RAG 要"检索相关文档片段",那它凭什么知道哪段文档和我的问题'相关'?
- 用关键词匹配?太弱了(同义词、换种说法就匹配不上)。真正的答案是 Embedding 

让机器理解"语义上的相似",而不是"字面上的相同"。

# Embedding 模型 - 通过向量算语意相似性

Embedding 就是把一段文本, 表示成一个定长的数字向量(通常是个数组，比如由1024个浮点数组成的数组)。语义越相近的文本,它们的向量在空间里离得越近。

于是"判断两段话意思像不像",就变成了"算两个向量离得近不近"

- token ID 只是"身份证号",纯粹的标识,没有含义
- embedding 向量是"语义坐标",位置本身就编码了含义

# 数据向量化

我们熟知的三维坐标系空间，用x轴、y轴、z轴来表示一个三维坐标系的空间。

数据向量化就像在坐标系中为每条数据生成一个带有方向的“箭头”（即向量），用一组浮点数来表示。但区别在于，AI 领域中的坐标系通常不是我们熟悉的三维（3D），而是高维的（比如 768维、1024维甚至更高）。
- 三维坐标系：只能容纳长、宽、高三个特征。
- 高维坐标系：每一个维度都代表数据的一个抽象特征，维度越高，能承载的数据细节和语义信息就越丰富。

在这样的高维空间中，判断两个向量是否接近（即语义是否相近），主要有两大派系的数学方法：

1. 看“方向”：余弦相似度（Cosine Similarity）
- 原理：它完全忽略向量的长度（比如文章的长短、数据的多少），只关注两个向量在空间中的指向是否一致
- 适用场景：非常适合文本语义匹配、用户兴趣推荐等场景。
2. 看“距离”：欧氏距离（Euclidean Distance）
- 原理：它计算的是两个向量在空间中的“直线距离”。距离越短（数值越小），说明两个点靠得越近，越相似。
- 适用场景：更适合图像特征匹配、空间位置分析等对绝对数值敏感的场景。

# 计算"两个向量是否相近" - 余弦相似度

衡量两个向量的语义接近度,最常用 余弦相似度(cosine similarity):算两个向量的夹角。
- 夹角越小(方向越一致)→ 余弦值越接近 1 → 表示语义越相似
- 夹角 90°(正交)→ 余弦值是0 → 表示语义无关
- 方向相反 → 余弦值接近-1 → 表示语义相反

### 计算余弦相似度

先来拿两个最简单的 3 维向量来当例子，手算一遍：

向量 A = [1.0, 2.0, 3.0]
向量 B = [4.0, 5.0, 6.0]

相似度 =（A点乘B）÷（A的模长 × B的模长）

1. 第一步：算“点积”（分子部分）
- 点积就是把两个向量对应位置的数字相乘，然后把乘积加起来。
    ```
    (1.0 × 4.0) + (2.0 × 5.0) + (3.0 × 6.0)
    ```
2. 第二步：算“模长”（分母部分）
- 模长就是算出每个向量自己的“长度”（也就是把每个数字平方后加起来，再开根号）。
    ```
    向量A的模长: √(1.0² + 2.0² + 3.0²) = √(1 + 4 + 9) = √14 ≈ 3.74
    向量B的模长: √(4.0² + 5.0² + 6.0²) = √(16 + 25 + 36) = √77 ≈ 8.77
    ```
3. 第三步：做除法（得出最终相似度）
- 用点积除以两个模长的乘积。
    ```
    32 ÷ (3.74 × 8.77) ≈ 0.975
    ```

真实业务里的向量通常是 768 维甚至 1024 维的，靠手算肯定不现实。在代码里，我们通常会用 ```NumPy```:

```py
import numpy as np

# 1. 准备两个向量（哪怕是一千个浮点数也没问题）
vec_a = np.array([1.0, 2.0, 3.0])
vec_b = np.array([4.0, 5.0, 6.0])

# 2. 一行代码算出点积（分子）
dot_product = np.dot(vec_a, vec_b) 

# 3. 一行代码算出模长（分母）
norm_a = np.linalg.norm(vec_a)
norm_b = np.linalg.norm(vec_b)

# 4. 最终算出相似度
cosine_similarity = dot_product / (norm_a * norm_b)

print(f"这两个向量的相似度是: {cosine_similarity:.4f}") 
# 输出: 0.9746
```

# 解决上下文大小限制问题

```
Embedding(把文本变成语义向量)
   ├─→ 语义搜索:把"问题"和"文档库"都变成向量,找最近的 → 这就是 RAG 的"检索"步骤
   ├─→ 记忆系统:把历史对话存成向量,需要时按语义召回相关片段(解决上下文的"窗口有限")
   └─→ 聚类/分类/去重:语义相近的自动聚到一起
```

一段对话迟早要到上窗口的上限，怎么办？
- 不把全部历史塞进去(塞不下),而是把历史会话存成向量,每轮只召回最相关的几段喂给模型。这就是"向量记忆"。

# 验证语意向量化

使用GLM为例，来为一段文本生成向量，这是返回的向量结果字段：

```json
{
  "model": "embedding-3",
  "data": [
    {
      "index": 123, // 结果下标,对应的输入文本在输入数组中的索引
      "embedding": [ // 向量化表征的数组，由于输入维度决定数量
        123
      ]
    }
  ],
  "usage": { // 用量
    "prompt_tokens": 123,
    "completion_tokens": 123,
    "total_tokens": 123
  }
}
```

GLM支持自定义的输出向量维度，Embedding-3 默认 2048，Embedding-2 固定 1024。Embedding-3 支持自定义，可选值：256、512、1024或2048。

### 验证示例：

生成向量和计算向量相似度：

```py
from zai import ZhipuAiClient
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import numpy as np

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
```

验证向量相似度和模拟RAG检索:

```py
from llm_fetcher import embeddings, calculate_similarity, EmbeddingData
import pandas as pd

sentences = [
    "我今天去公园散步了", # A
    "我在公园走了走,很惬意", # A' (和A语义近,但用词不同)
    "Swift 是一门编程语言", # B (和A完全无关)
    "苹果公司开发了 Swift", # B' (和B相关)
    "苹果公司开发了 OC语言",
    "Swift语言是由苹果公司开发",
    "苹果公司开发的Swift语言非常安全",
]
embedding_messages: list[EmbeddingData] = []

def vector(inputs: list[str]) -> list[EmbeddingData]:
    return embeddings(inputs=inputs)

def search(prompt: str, top_k: int = 3) -> list[tuple[str, float]]:
    """语义检索:返回最相似的 top_k 条 (文本, 相似度分数)"""
    # 先生成待检索的向量
    prompt_vec = embeddings(inputs=[prompt])[0].embedding
    # 和“向量数据库”进行比较相似度，生成相似度列表
    scores: list[tuple[str, float]] = [
        (sentences[e.index], calculate_similarity(prompt_vec, e.embedding))
        for e in embedding_messages
    ]
    # 按相似度降序,取前 top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main():
    # 生成向量数据库
    global embedding_messages
    embedding_messages = vector(inputs=sentences)
    print(f"向量数据库: {len(embedding_messages)}")

    # 比较两个向量的相似度
    n = len(sentences)
    matrixs = [[calculate_similarity(embedding_messages[i].embedding,
                                embedding_messages[j].embedding)
                for j in range(n)] for i in range(n)]

    df = pd.DataFrame(data=matrixs, index=sentences, columns=range(n))
    print(df.round(3))

    # RAG语意检索
    texts = search(prompt="哪门语言是苹果做的?")
    print(f"\nRAG检索出结果：{texts}")

if __name__ == "__main__":
    main()
```

输出结果：
```
向量数据库: 7
                        0      1      2      3      4      5      6
我今天去公园散步了           1.000  0.878  0.482  0.449  0.437  0.444  0.410
我在公园走了走,很惬意         0.878  1.000  0.485  0.458  0.455  0.453  0.452
Swift 是一门编程语言       0.482  0.485  1.000  0.868  0.687  0.892  0.779
苹果公司开发了 Swift       0.449  0.458  0.868  1.000  0.809  0.926  0.841
苹果公司开发了 OC语言        0.437  0.455  0.687  0.809  1.000  0.786  0.724
Swift语言是由苹果公司开发     0.444  0.453  0.892  0.926  0.786  1.000  0.852
苹果公司开发的Swift语言非常安全  0.410  0.452  0.779  0.841  0.724  0.852  1.000

RAG检索出结果：[('苹果公司开发了 OC语言', np.float64(0.6909331508195076)), ('Swift语言是由苹果公司开发', np.float64(0.6797630454007655)), ('苹果公司开发了 Swift', np.float64(0.6414242031717567))]
```

# Anki卡片

```
Q: Embedding 是什么?
把文本映射成定长数字向量,语义越相近的文本向量在空间里越近

Q: token ID 和 embedding 向量的本质区别?
token ID 是"身份证号"(纯标识,无语义关系);embedding 是"语义坐标"(位置本身编码含义,相近的近)

Q: 怎么衡量两个 embedding 的语义相似度?
余弦相似度,算两向量夹角;越接近1越相似,0无关,-1相反

Q: 为什么用余弦(夹角)而非直线距离?
关心语义方向而非向量长度;夹角剔除长度干扰,只看语义指向

Q: Embedding 是哪些技术的地基?
语义搜索(RAG检索)、向量记忆、聚类/分类/去重

Q: 语义搜索比关键词匹配强在哪?
它匹配"意思"而非"字面";用词不同但意思相近也能召回(如"散步"匹配"走了走")

Q: RAG 的"检索"步骤本质是什么?
把query和文档库都转成向量,用余弦相似度找最相关的片段召回

Q: 调 embedding API 的工程注意点?
用批量接口一次传多句,别循环单条调用(慢且费)

Q: RAG 检索返回结果,用"固定阈值"还是"Top-K"
主流用 Top-K(返回最相似的前K条),固定阈值脆弱(依赖模型/领域);常 Top-K + 下限阈值结合

Q: 向量检索召回的片段,等于问题的正确答案吗?
不等于。embedding 只找"语义最相近"的(如问"苹果的语言"会把OC排在Swift前),哪个真正回答问题要靠后续LLM判断。所以RAG是"检索+生成"两步,缺一不可。
```

# 模型生态地图调研

数据截止到：2026/05

| 模型 | 厂商 | 定位/强项 | 开源 | 大致价格档 | 上下文窗口 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Claude | Anthropic | 代码工程王、长文本专家、安全合规 | 否（完全封闭） | 较高（旗舰Opus系列定价最贵） | 1M Token |
| GPT | OpenAI | Agent编程全能、全流程自动化、多模态交互 | 否（闭源） | 高（API调用成本较高） | 1M Token |
| Gemini | Google | 科学推理第一、原生多模态、生态联动 | 否（闭源） | 中等 | 10M Token |
| DeepSeek | 深度求索 | 性价比之王、数学与理科强者 | 是（开源+闭源） | 极低（极致性价比） | 未明确提及具体上限 |
| Qwen/通义千问 | 阿里 | 中文能力顶尖、开源生态最全 | 是（开源+闭源） | 较低（高性价比） | 1M Token |
| GLM | 智谱AI | 企业级逻辑推理专家、中文注释质量高 | 是（开源编程旗舰） | 较低 | 未明确提及具体上限 |
| Llama | Meta | 开源界性能黑马、私有化部署首选 | 是（免费商用） | 免费（开源自行部署） | 扩展至128K Token |
