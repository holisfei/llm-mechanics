# 模型预测机制总览

```
预训练(Pretrain) → 得到base 模型:只会"猜下一个 token",像超级自动补全,不听话
        ↓ 通过 SFT(监督微调):喂"指令→好答案"的例子,教它"被问要回答"
        ↓ 通过 RLHF(人类反馈强化学习):用人类偏好打分,对齐成"有用/无害/诚实"
   = 最终 chat 模型:听话、会对话、像个助手
```

# 自回归生成(autoregressive)

我们已经知道了：模型只认识数字，是一级文本的补全模式/概率倾向，它将一段文本切分成一个个token，对每个token做编码ID，再对ID做计算和预测

所以，模型本质只干一件事：给定前面已有的一串token，预测下一个token最可能是什么，是基于概率的预测行为：
1. 你输入 "今天天气真"
2. 模型算出:下一个 token 是"好"的概率 60%、"热"的概率 20%、"差"的概率 10%……(它对整个词表几万个 token 都算一个概率)
3. 按某种策略选一个(比如选概率最高的"好")
4. 现在输入变成"今天天气真好",把这个新串重新喂回去,再预测下一个 token
5. 循环,直到生成一个特殊的"结束"token

这个就是自回归生成(autoregressive)
- 它没有“我在回答问题”这种意识，只是在补全一段看上去应该怎么去接下去的文本
- 这也解释了模型幻觉的原因，它补全了“看起来正确”的内容，而不是补全了“基于事实”的内容
- 这也解释了base模型为什么不听话

# 模型的流式输出

模型是一个 token 一个 token 生成的,流式输出能让我们看到这个过程

这里以GLM的流式输出为示例:

```
ChatCompletionChunk (
    id ='20260525103320b44e019a23be4f74', 
    choices=[
        Choice(
            delta=ChoiceDelta(
                content='（', 
                role='assistant', 
                reasoning_content=None, 
                tool_calls=None, 
                audio=None
            ), 
            finish_reason=None, 
            index=0
        )
    ],     
    created=1779676400, 
    model='glm-4.5-air', 
    usage=None, 
    extra_json=None, 
    object='chat.completion.chunk'
)
```

我们看到每一段文本/一个文字/符号都被包装成了一个独立的```ChatCompletionChunk```对象
- ```finish_reason='stop'```的时候，表示流结束的```chunk```
- ```choices[0].delta.content```是每个```chunk```的内容

在流结束的```chunk```，会返回本次请求中token的使用统计情况：
```
usage=CompletionUsage(
    prompt_tokens=9, 
    prompt_tokens_details=PromptTokensDetails(
        cached_tokens=4
    ), 
    completion_tokens=1278, 
    completion_tokens_details=None, 
    total_tokens=1287
)
```

其中```prompt_tokens_details```,它的意思是:这次请求的输入里,有 4 个 token 命中了缓存(之前算过、不用重算),这部分通常更便宜甚至免费。

我们用一段代码来看下```token```和```chunk```的大致关系:
```py
    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {'role': 'user', 'content': promt},
        ],
        stream=True
    )
    
    for chunk in response:
        chunk_count = chunk_count + 1
        if chunk.usage: # 流结束 统计token用量
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            total_tokens = chunk.usage.total_tokens
            print(f"用量：token:{total_tokens}(提示词:{prompt_tokens} 回复:{completion_tokens}), chunk:{chunk_count}")
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```
结果： ```用量：token:1477(提示词:9 回复:1468), chunk:1452```

我们看到```token```和```chunk```不是1:1的关系，```token```用量大于```chunk```，也就是说一个```chunk```可能包含多个```token```

# 模型是怎么被"驯化"的

底层模型通过3个阶段来一步步让其”更听话“

### 阶段1：预训练(Pretrain) → 得到 base 模型

模型公司会将整个互联网海量的文本和知识都给模型，让它基于概率去预测下一个token是什么，它学完后拥有了海量的知识和语言能力，但是它只会补全文本(预测)，不知道用户是否下了指令、问了问题
- 比如，你对模型说"帮我写封邮件，它可能会回复你”"帮我写封邮件用英语怎么说?帮我写封信呢?"“，它没有回答你真正的问题，它基于海量的文本语料补全了你的问题。

这就是最开始的base模型：拥有了海量知识，只会补全出一个'回答'形态的文本"

### 阶段2：通过 SFT → 让模型学会"被问的时候要回答"

SFT：Supervised Fine-Tuning, 监督微调

人类准备了大量的高质量配对文本"收到指令/问题→理想回答"，拿这些数据继续训练 base 模型，模型拥有了另外的能力：当收到指令/问题的时候，应该去找到问题的理想答案做补全，而不是基于问题做补全

在这个阶段，模型已经可以处理的用户问题了。

### 阶段3：RLHF → 对齐成"有用/无害/诚实"

RLHF：人类反馈强化学习

SFT之后，模型可以回答问题，但是回答的好不好，是否安全，是否需要过滤有害的问题请求，还需要继续打磨。

所以，RLHF 的做法是: 让模型对同一个问题生成多个答案，由人类或者另外一个模型(一个学了人类偏好的"奖励模型")给生成的多个答案进行基于多维度的打分，再强化学习让模型倾向于生成高分的答案。

所以，chat 模型比 base 模型好用, 是因为它在 base(通过海量知识会补全)的基础上,经过 SFT(模型学会"对于指令/问题要回答") 和 RLHF(模型学会"答得有用、无害、诚实"),被"驯化"成了一个听话的助手。

这也解释了以下问题：
- system prompt（系统提示词）为什么有效? 因为模型在 SFT/RLHF 阶段,被大量训练成要 "遵循 system 角色给定的设定"
- prompt engineering（提示词工程），因为模型是被人类基于概率/打分等机制调教出来的，不是确定的逻辑。

# Anki卡片

```
Q: 模型生成的本质动作是什么?
给定前面所有 token,预测下一个 token 的概率分布,选一个,再循环

Q: 什么是自回归(autoregressive)生成?
把自己生成的 token 拼回输入,作为下一次预测的依据,循环往复

Q: "模型是超级自动补全"这句话,对在哪、易误解在哪?
对在数学本质是 P(下一个token|前文);易误解在它没有"在回答问题"的意识,只是补全最合理的文本

Q: 预训练(pretrain)得到的 base 模型有什么特点?
知识渊博、语言能力强,但只会补全、不听指令

Q: SFT 解决了什么问题?
用"指令→理想回答"配对训练,让模型学会"被问要回答"而非续写更多问题

Q: RLHF 解决了什么问题?
用人类偏好打分排序 + 强化学习,把模型对齐成"有用、无害、诚实"

Q: system prompt 为什么有效?
模型在 SFT/RLHF 中被训练成遵循 system 角色设定,所以会照设定行事

Q: chat 模型"理解"了用户意图吗?
没有。它和 base 一样不理解,只是被 SFT/RLHF 训练成"把指令/问题补全成回答的形态"。全程是模式匹配,不是理解。

Q: 流式返回里的 cached_tokens 是什么?对 Agent 有何意义?
命中 prompt 缓存(KV Cache)的输入 token 数,通常更便宜;Agent 每轮重发长 system+历史,靠缓存大幅降本。
```

