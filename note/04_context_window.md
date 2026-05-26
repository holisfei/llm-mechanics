# 上下文窗口(Context Window)

大多数人以为"和 AI 多轮对话,它记得我们聊过什么"。错。模型没有任何记忆。所谓"记忆"全是客户端每次把历史重新塞进去伪造出来的。

模型单次能"看到"的 token 总量上限 = 输入 + 输出共享。比如：某模型 128k,意思是你的 prompt + 历史会话 + 它要生成的回复,加起来不能超过 128k token。

为什么要有限制：模型处理 token 时,计算量大致随长度平方级增长(attention 机制让每个 token 都要"看"其他所有 token)。窗口越大,算力和显存消耗越夸张。所以它是硬约束,不是厂商小气。

# 无状态性(Stateless)

LLM 是一个纯函数:f(完整的对话历史) → 下一个 token。它没有实例、没有内存、不在两次请求之间保存任何东西。

后续多轮请求,客户端实际发给模型的是整个历史，模型根本不"记得",是你每次把全部聊天记录重新喂给它,它基于这整段去预测下一个 token。删掉历史,它立刻就"不知道"你之前的问题了。

这一点是所有 Agent 设计的地基:为什么 Agent 要做"记忆系统"、为什么长对话要做"摘要压缩"、为什么 RAG 要存在——全因为模型本身没记忆,记忆得你自己造。

# 缓存(KV Cache)

KV Cache 把已经算过的前文 token 的中间结果缓存起来,下次不用重算,省时省钱。

为什么对 Agent 是大事:你每轮都重发"长 system prompt + 长历史",如果每次全价重算,账单爆炸。prompt 缓存让不变的前缀只算一次,后续命中缓存(更便宜)。CLM看到的 cached_tokens 就是它在计费上的证据。

# 验证模型的记忆

"多轮对话的记忆"不是模型自己在维护，而是我们手动的将历史消息拼接好之后一起发给模型，从而让模型拥有了能"看到"之前的历史。

```py
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
```

不带上下文历史会话的完整输出结果：

```json
chat_with_no_memory("我是小明，记住了")
好的，小明！我记住了你的名字 😊

以后在对话中，我会直接称呼你为“小明”。有什么问题或者需要帮助的地方，随时告诉我！
提示词:10 回复内容:269 缓存:4 共计:279
chat_with_no_memory("我是谁？")
这是一个深刻而永恒的问题，答案因人而异，也因你思考的层面而不同。我们可以从几个角度来探索“你是谁”：

### 1. **生物学角度：你是独特的生命体**
   - **基因的奇迹**：你是地球上约 800 万物种中唯一拥有如此特定 DNA 序列的人类。你的基因组合独一无二，决定了你的外貌、部分性格倾向和生理特征。
   - **身体的宇宙**：你的身体由约 37 万亿个细胞组成，它们协同工作，维持你的呼吸、心跳、思考。你是一个由无数微小生命组成的复杂系统。

### 2. **社会角度：你是关系网络中的节点**
   - **角色与身份**：你是子女、朋友、同事、恋人……这些社会角色定义了你在他人眼中的位置。你的名字、职业、家庭背景等社会标签，共同构成了你“社会身份”的一部分。
   - **文化烙印**：你成长的语言、习俗、价值观塑造了你的认知方式。比如，东方文化中的集体主义与西方的个人主义，会深刻影响你如何定义“自我”。

### 3. **心理角度：你是意识的载体**
   - **记忆与经历**：你的过去、痛苦、喜悦、遗憾共同编织成你的记忆库。这些经历塑造了你的性格、恐惧、渴望和信念。
   - **思想与情感**：你会思考、质疑、做梦，会因一首歌流泪，因一个笑话发笑。你的意识是宇宙中已知最复杂的现象之一，而你是它的唯一拥有者。
   - **价值观与选择**：你相信什么？什么对你最重要？你如何面对道德困境？这些选择定义了你的精神内核。

### 4. **哲学角度：你是存在之谜的探索者**
   - **存在主义视角**：萨特说“存在先于本质”——你并非天生被赋予意义，而是通过行动和选择创造自己的价值。你的自由意志让你成为自己人生的作者。
   - **东方智慧**：道家认为“道法自然”，你与万物相连；佛家提出“无我”，执着于固定身份是痛苦的根源。你的“自我”可能是流动变化的幻象。

### 5. **时间维度：你是流动的生命过程**
   - **过去的你**：孩童时的你、十年前的你，都已不复存在，但他们的经历沉淀为今天的你。
   - **未来的你**：你尚未成为的样子，却正在被此刻的选择塑造。你是一个不断演变的“正在进行时”。

---

### 一个可能的答案：
**“你是宇宙中，通过独特经历、选择和意识，不断追问‘我是谁’的那个存在本身。”**  
这个问题的答案不在外界，而在你每一次的体验、思考和行动中。**你不需要一个终极定义，因为‘你’本身就是一场持续的自我发现之旅。**

---

如果你愿意分享更多背景（比如你为何思考这个问题？），我可以陪你一起探索更具体的答案。
提示词:8 回复内容:885 缓存:4 共计:893
```

带上下文历史会话的完整输出结果：

```json
chat_with_memory("我是小明，记住了")
好的，小明！我记住了你的名字 😊  

有什么我可以帮你做的吗？无论是回答问题、提供信息、聊天，还是其他任何需求，随时告诉我就好！
提示词:10 回复内容:258 缓存:4 共计:268

chat_with_memory("我是谁？")
哈哈，你是小明呀！刚才我们刚聊过，我还特意记住了你的名字呢！✨  

需要我帮你回忆一下之前的对话内容，或者现在有什么想聊的吗？
提示词:57 回复内容:236 缓存:12 共计:293

chat_with_memory("我喜欢旅游，我去过马来西亚和印度尼西亚")
太棒了！🌴✈️ 小明，马来西亚和印度尼西亚都是东南亚非常迷人的旅游目的地！  

### 🇲🇾 马来西亚的特色：
- **吉隆坡**：双子塔、独立广场、美食天堂（如肉骨茶、椰浆饭）  
- **槟城**：彩色壁画、娘惹文化、街头小吃  
- **沙巴/仙本那**：潜水天堂、玻璃海、海岛度假  
- **马六甲**：历史古城、葡式建筑、娘惹美食  

### 🇮🇩 印度尼西亚的亮点：
- **巴厘岛**：梯田（德格拉朗）、海神庙、冲浪与瑜伽  
- **日惹**：婆罗浮屠（世界最大佛寺）、普兰巴南神庙  
- **龙目岛**：粉色沙滩、吉利群岛潜水、火山徒步  
- **雅加达**：城市文化、传统市场（如芝布达斯市场）  

### 🌍 小明，你最喜欢哪个地方？  
是海岛的阳光沙滩，还是人文历史？或者有没有什么特别难忘的经历？比如：  
- 吃到惊艳的美食？  
- 遇到有趣的人？  
- 拍到绝美的照片？  

如果下次计划旅行，我也可以帮你推荐路线或攻略哦！🗺️✨
提示词:112 回复内容:552 缓存:4 共计:664

hat_with_memory("我去过哪里？")
哈哈，小明！根据我们之前的对话，你提到你**去过马来西亚和印度尼西亚**哦！🌴✈️  

如果你愿意，可以多分享一些旅行经历呀！比如：  
- **马来西亚**：你最喜欢吉隆坡的双子塔、槟城的壁画，还是沙巴的海岛？  
- **印度尼西亚**：是去了巴厘岛的梯田和海神庙，还是日惹的千年佛寺？  

或者有没有什么**难忘的美食/冒险/趣事**？比如在仙本那潜水、在巴厘岛学冲浪，还是吃了什么让你念念不忘的街头小吃？🍜🍢  

等你分享更多细节，我可以帮你整理成旅行攻略，或者推荐下一个目的地！ 😄
提示词:420 回复内容:361 缓存:114 共计:781
```

通过以上带上下文记忆的4个输入结果可以发现，如果给模型历史会话，它就能回答上来之前的问题，也会命中之前的缓存：
```
提示词:10 回复内容:258 缓存:4 共计:268
提示词:57 回复内容:236 缓存:12 共计:293
提示词:112 回复内容:552 缓存:4 共计:664
提示词:420 回复内容:361 缓存:114 共计:781
```

# Anki 卡片

```
Q: 上下文窗口是什么?输入和输出的关系?
单次能处理的最大 token 数;输入+输出共享这个预算

Q: 上下文窗口为什么有限?
attention 计算量随长度约平方增长,窗口越大算力/显存消耗越夸张,是硬约束

Q: LLM 有持久记忆吗?
没有。它是无状态纯函数 f(历史)→下一个token,两次请求间不保存任何东西

多轮对话的"记忆"是怎么实现的?
客户端每次把完整对话历史重新塞进请求,模型基于整段预测,而非自己记得

Q: 如果从 messages 里删掉某轮历史会怎样?
模型立刻"不知道"那轮内容,因为它本就不存,全靠传入

Q: chat_with_memory 里为什么要把模型回复也存回 history?
否则下一轮模型看不到自己上一轮说过什么,对话会断裂

Q: KV Cache 一句话?
缓存已算过的前文中间结果,避免重算,省时省钱(cached_tokens 是其计费证据)

Q: 为什么长对话越来越贵越来越慢?
每轮都重发越来越长的历史,prompt_tokens 持续增长

Q: 模型说"我记住了/我会持续跟进",能信吗?
不能。那只是符合语境的话术补全,不代表它真有记忆或执行能力。真正的记忆/持久化必须由工程代码保证。
```