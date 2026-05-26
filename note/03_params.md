# 采样参数对模型的影响(GLM为例)

对于 开放性prompt 和 事实性/确定性prompt，参数带来的影响是不同的：
- temperature 和 top_p 对于开放性的 prompt，影响的是生成结果的多样性
- do_sample，对于事实性/确定性prompt，官方建议用False，表示生成结果遵循事实采样

模型对下一个 token 算出了一个概率分布,比如:
```
好 60% | 热 20% | 差 10% | 棒 5% | ...(词表里几万个,大多概率极低)
```

怎么从这个分布里挑一个 token 出来? 这就是"采样策略",由下面几个采样参数控制。

### temperature(温度)

调整"分布的陡峭程度"，改变这个分布的"陡峭/平缓"，它不挑词,只改概率分布的形状,改完之后才去随机抽。

- temperature = 0:不抽了,直接选最高的 → 必出"北京"。
- temperature 低(0.2):把分布拉得更陡,变成大概:
    ```
    北京 95% | 上海 3% | 南京 1.5% | ...
    ```
- temperature 高(1.0):把分布压平,变成大概:
    ```
    北京 55% | 上海 18% | 南京 12% | 一 8% | 其余 7%
    ```

### top_p(核采样 / nucleus sampling)

调整"候选池大小(按累积概率)"，它不改概率,只决定"哪些词有资格参与抽奖":从高到低累加概率,够到 p值 就停,后面的全部丢弃。

top_p 不改概率本身,只用累加概率来圈定候选范围。概率分布的形状没变,变的是"哪些词有资格进抽奖箱

关键澄清(很多人搞错):top_p=0.2 不是"取 20% 数量的词",而是"只保留累积概率到 20% 的高概率词";分布很尖锐时,候选池可能只有 1-2 个词

- top_p = 0.8:从高往低加,北京 80% 就已经≥80%了 → 候选池里只剩"北京"一个,必出北京。
- top_p = 0.9:北京80% + 上海8% = 88% < 90%,再加南京 93% ≥90% → 候选池 = {北京, 上海, 南京},在这三个里按概率抽。
- top_p = 1.0:全词表都有资格(包括那几万个冷门词)→ 最多样,也最容易蹦出奇怪的词。

 ### top_k(很多 API 默认不暴露)

- top_k 是保留"固定数量 K 个"最高分的词;
- top_p 是保留"累计概率范围"。区别就在"固定个数" vs "动态按概率"。GLM 的对话补全主要给你 temperature 和 top_p。

### temperature 和 top_p:为什么"二选一"?

两个参数都在控制"输出的随机性/多样性",只是机制不同(一个调分布陡峭度,一个调候选池范围)。同时调两个,效果会叠加纠缠,难以判断是谁起的作用,所以官方建议二选一

### do_sample 确定性开关

官方推荐代码/翻译用 do_sample=False

do_sample=False 和"事实/正确"没有任何关系。它只是让模型每步都选概率最高的那个 token(贪心),从而追求可复现/确定性,不是追求"对"

# 用模型做实验(GLM为例)

用同一个提示词，输入不同的参数，对比结果

```py
from dotenv import load_dotenv
import zai
from zai import ZhipuAiClient
import pandas as pd
import asyncio

load_dotenv()
client = ZhipuAiClient()

def fetch(params: dict) -> str:
    try:
        response = client.chat.completions.create(**params) # 异步 asyncCompletions
        content = response.choices[0].message.content
        # print(f"结果:{json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")
        return content
    except zai.core.APIStatusError as err:
        e = f"API 状态错误: {err}"
        print(e)
        return e
    except zai.core.APITimeoutError as err:
        e = f"请求超时: {err}"
        print(e)
        return e
    except Exception as err:
        e = f"其他错误: {err}"
        print(e)
        return e
 
async def run_experiment(
    prompt: str, 
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    count: int = 3,
    max_token: int = 1500,
    is_async: bool = False
) -> list[str]:
    """同一参数跑 n 次,返回 n 个输出文本。"""

    kwargs: dict = {
        "model": "glm-4.5-air",
        "messages": [
            {'role': 'user', 'content': prompt},
        ],
        "stream": False,
        "max_tokens": max_token # 不能超出给定的token数量
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if do_sample is not None:
        kwargs["do_sample"] = do_sample

     # asyncio.to_thread 将一个同步函数加入异步任务
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(asyncio.to_thread(fetch, kwargs)) for _ in range(count)]
    
    
    return [task.result() for task in tasks]

async def run_params(
    temperature_list: list[float] | None = None, 
    top_p_list: list[float] | None = None, 
    do_sample: bool | None = None,
    prompt: str = "用一句话描述大海"
):
    # temperature
    results: dict[float:list[str]] = {}

    if temperature_list is not None:
        for t in temperature_list:
            result = await run_experiment(prompt, temperature=t, do_sample=do_sample)
            results[f"{t}-do_sample_{do_sample}" if do_sample is not None else t] = result
    if top_p_list is not None:
        for p in top_p_list:
            result = await run_experiment(prompt, top_p=p)
            results[p] = result
    if do_sample is not None:
        result = await run_experiment(prompt, do_sample=do_sample)
        results[f"do_sample_{do_sample}"] = result

    # 表格, 字典的键（key）作为列名，值（value）作为列的数据：
    df = pd.DataFrame(results)
    df.insert(loc=0, column="次数", value=[f"第{i+1}次" for i in range(len(df))])
    print(df)

async def main():
    await run_params(temperature_list=[0.0, 0.3, 0.7, 1.0])
    await run_params(top_p_list=[0.1, 0.5, 1.0])
    await run_params(do_sample=True)
    await run_params(do_sample=False, temperature_list=[1.0])

if __name__ == "__main__":
    asyncio.run(main())
```

### temperature 结果

temperature = [0.0, 0.3, 0.7, 1.0]

```
    次数                                        0.0  \
0  第1次                大海是无垠的蔚蓝，潮汐呼吸间，孕育着星辰与浪花的永恒。   
1  第2次  大海是蔚蓝的永恒呼吸，潮汐是它的心跳，浪花是它的絮语，深邃处容纳着星辰与万千故事。   
2  第3次                大海是无垠的蔚蓝，以浪花的笔触书写着深邃与星辰的永恒。   

                               0.3                                 0.7  \
0    大海是无垠的蔚蓝，时而温柔地托起浪花，时而深沉地容纳百川。      大海是地球的蓝色呼吸，潮起时拥抱海岸，落时又藏进深蓝的梦里。   
1        大海是蔚蓝的深渊，潮汐起伏间吟唱着永恒的生命之歌。  大海是蔚蓝的胸怀，浪花翻涌着永恒的呼吸，深邃里藏着星辰与万物的回响。   
2  大海是蔚蓝的胸怀，以潮汐的呼吸吞吐日月，用深邃的怀抱藏纳星辰。         大海是蔚蓝的呼吸，吞吐着日月星辰与万千浪花的永恒诗篇。   

                                                      1.0  
0                             大海是蔚蓝辽阔的胸怀，潮汐起伏间藏着生命的深邃与包容。  
1                      大海是流动的诗卷，以蔚蓝为笔、浪花为墨，写下潮汐的永恒与深蓝的无垠。  
2  大海是浩瀚的蔚蓝摇篮，以浪花的呼吸拥吻天地，又将潮汐的絮语、星辰的倒影和生命的微光，都织进深不见底的温柔里。  
```

### top_p 结果

top_p = [0.1, 0.5, 1.0]

```
 次数                                    0.1  \
0  第1次  大海是流动的蔚蓝，以浪花的低语拥抱天地，用深邃的胸怀藏纳星辰与未知的远方。   
1  第2次  大海是流动的蔚蓝，以浪花的低语拥抱天地，用深邃的胸怀藏纳星辰与未知的远方。   
2  第3次        大海是蔚蓝的永恒，以潮汐为呼吸，托起日月星辰，也容纳万千生命。   

                                  0.5  \
0  大海是蔚蓝的胸怀，以潮汐的呼吸吞吐星辰，用无垠的深邃容纳浪花与远方。   
1           大海是蔚蓝的胸怀，以潮汐吞吐日月，用深邃包容万物。   
2       大海是无垠的蔚蓝，翻涌着不息的浪潮，深邃里藏着星辰与永恒。   

                                        1.0  
0  大海是天地间流动的蔚蓝诗篇，以浪花为笔，潮汐为韵，书写着永恒的深邃与生命的呼吸。  
1              大海是无垠的蓝绸，裹着浪花的呼吸，藏着星辰与深海的絮语。  
2                   大海是蔚蓝的深渊，盛着潮汐的呼吸与星辰的倒影。  
```

### do_sample 结果

do_sample = True：

```
    次数                  do_sample_True
0  第1次  大海是广袤的蔚蓝波涛，以永恒的呼吸吞吐着天地间的深邃与包容。
1  第2次         大海是流动的蔚蓝，裹挟着星辰与浪花的永恒呼吸。
2  第3次            大海是蔚蓝的摇篮，托起日月，也孕育生命。
```

do_sample = False，temperature = 1.0：

```
   次数                              1.0-do_sample_False
0  第1次                  大海是无垠的蔚蓝怀抱，裹着浪花的呼吸，向天际铺展着生命的诗行。
1  第2次                       大海是天地揉碎的蓝，潮起时奔涌星河，落日里熔金万顷。
2  第3次  大海是蔚蓝的永恒诗篇，潮汐是它起伏的呼吸，浪花是它写给天空的信，藏着星辰与万千生命的深邃故事。
```

# Anki 卡片

```
Q: 采样参数(temperature/top_p)作用在生成的哪一步?
模型已算出下一个 token 的概率分布后,如何从分布中挑选那一步

Q: temperature 的直觉作用?
调整概率分布的陡峭程度;低→陡峭→确定保守,高→平缓→多样有创意

Q: temperature=0 大致等于什么策略?
贪心,几乎总选概率最高的 token(但工程上未必 100% 可复现)

Q: top_p(核采样)是什么?
从高到低累积 token 概率到阈值 P,只在这批候选里采样

Q: top_p=0.2 的正确理解?
保留"累积概率到 20%"的高概率词,不是"数量前 20%";分布尖锐时可能只剩 1-2 个词

Q: top_k 和 top_p 的区别?
A: top_k 保留固定 K 个最高分词;top_p 按累积概率动态决定候选池大小

Q: 为什么不建议同时调 temperature 和 top_p?
两者都控制随机性,机制不同,同时调会纠缠叠加、难以归因;官方建议二选一

Q: GLM 的 do_sample=False 有什么效果?适合什么任务?
关闭随机采样,总选最高概率词、输出确定可复现;适合代码生成、翻译等需一致性的任务

Q: do_sample=False / temperature=0 追求的是"正确"还是"确定"
 确定(可复现),不是正确。模型可以稳定地输出错误答案。确定性≠正确性。
```