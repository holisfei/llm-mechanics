# 理解幻觉的本质

模型在做"概率上最合理的续写",不是"事实检索"。

它的目标是生成"读起来对"的文本,而不是"事实上对"的文本。而且它的架构里,没有任何一个模块负责"核查真假",也没有"我不知道"这个内置开关。

```json
输入文本→切分token→编码ID
模型只会"预测下一个最可能的 token"(补全),不理解、不检索
模型无状态、无记忆,知识只来自训练时"压缩"进参数的东西
  ↓
当你问的东西,它参数里没有/记不清时——
    它不会说"我不知道"(没这个开关),
    而是继续干它唯一会干的事:补全出"看起来最合理"的 token,
    = 一本正经地编。
```

模型为什么会发生幻觉?
- 机制层面(最根本):它本质是"预测下一个 token",目标是生成流畅合理的文本,架构里缺乏事实验证机制。流畅 ≠ 真实
- 知识层面:训练数据有知识缺口、噪声、过时信息;而且知识是被"压缩"进参数的,长尾、冷门的事实它记不清或没见过。还有知识截止日期——截止后发生的事它根本不知道
- 采样层面(接 Day3):高 temperature 让低频 token 更易被选中,加剧幻觉。这就是为什么需要事实的任务要用低温/do_sample=False
- 架构层面:注意力缺陷、单向自回归的固有局限等

"模型说得这么自信、这么有条理,认为模型说的都对"这是错误的认知

自信和条理,恰恰是它最擅长的(它就是被训练来生成流畅文本的)。流畅度、自信度,和正确性完全没有关系。 一个编造的答案,可以和正确答案一样流畅自信

# 验证模型的幻觉

```py
from llm_fetcher import fetch, ROLE, ContentModel
import asyncio
import pandas as pd

pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.max_colwidth', None)  # 单元格内容不截断
pd.set_option('display.width', None)         # 自动适应终端宽度

def chat(
    input: str, 
    temperature: float | None = None, 
    do_sample: bool | None = None
) -> ContentModel:
    paras = {
        "model": "glm-4.5-air",
        "messages": [
            {"role": ROLE.user, "content": input},
        ],
        "stream": False,
    }
    if temperature is not None:
        paras["temperature"] = temperature
    if do_sample is not None:
        paras["do_sample"] = do_sample

    result = fetch(paras)

    return result
 
async def main():
     # 引导 LLM 乱输出，事实不存在的提示词
    prompts = [
        # "请介绍一下作家张伟立 2023 年出版的小说《雾港深处》的剧情",
        "Swift 6 里新增的 @AsyncReentrant 属性包装器怎么用?",
        # "神州23号飞船发射的时候，当时央视直播间的主持人是谁？"
        # "《红楼梦》第 88 回里林黛玉对贾宝玉说的那句关于茶的诗是什么?",
    ]
    
    const = "如果你不确定或不知道,请直接说'我不知道',不要编造。"

    async with asyncio.TaskGroup() as tg:
        tasks_dict: dict[str, list] = {}
        for p in prompts:
            # 不带约束提示词
            task1 = tg.create_task(asyncio.to_thread(chat, input=p, temperature=0.0))
            # 带约束提示词
            task2 = tg.create_task(asyncio.to_thread(chat, input=f"{p},{const}", temperature=0.0))
            # 收集tasks
            tasks_dict[p] = [task1, task2]
         
    datas: dict[str, list[str]] = {}
    for p, tasks in tasks_dict.items():
        res1: ContentModel = tasks[0].result()
        res2: ContentModel = tasks[1].result()
        datas[p] = [res1.content, res2.content]

    df = pd.DataFrame(datas)
    df.insert(loc=0, column="场景", value=["不带约束", "带约束"])
    print(df)

    
if __name__ == "__main__":
    asyncio.run(main())
```