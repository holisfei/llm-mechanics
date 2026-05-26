from llm_fetcher import embeddings, calculate_similarity, EmbeddingData
import pandas as pd

# 强制所有列名左对齐（默认数字右对齐，文本左对齐，统一后视觉更整齐）
pd.set_option('display.colheader_justify', 'left')
# 【关键】如果数据包含中文，必须开启此项，否则中文字符宽度计算错误会导致严重错位
pd.set_option('display.unicode.east_asian_width', True)
# 取消整体显示宽度限制，防止表格被强行换行破坏结构
pd.set_option('display.width', None) 

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

    # 两两比较向量的相似度
    n = len(sentences)
    # 二维数组矩阵
    matrixs: list[list[float]] = []
    for i in range(n):
        temp_s: list[float] = []
        for j in range(n):
            s = calculate_similarity(embedding_messages[i].embedding,
                                     embedding_messages[j].embedding)
            temp_s.append(s)
        matrixs.append(temp_s)

    # columns: 第一行，index：第一列，data:每一行的数据
    df = pd.DataFrame(data=matrixs, index=sentences, columns=range(n))
    print(df.round(3))

    # RAG语意检索
    texts = search(prompt="哪门语言是苹果做的?")
    print(f"\nRAG检索出结果：{texts}")

if __name__ == "__main__":
    main()