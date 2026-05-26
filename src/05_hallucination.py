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
    params = {
        "model": "glm-4.5-air",
        "messages": [
            {"role": ROLE.user, "content": input},
        ],
        "stream": False,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if do_sample is not None:
        params["do_sample"] = do_sample

    result = fetch(params)

    return result
 
async def main():
     # 引导 LLM 乱输出，事实不存在的提示词
    prompts = [
        # "请介绍一下作家张伟立 2023 年出版的小说《雾港深处》的剧情",
        "Swift 6 里新增的 @AsyncReentrant 属性包装器怎么用?",
        # "神州23号飞船发射的时候，当时央视直播间的主持人是谁？"
        # "《红楼梦》第 88 回里林黛玉对贾宝玉说的那句关于茶的诗是什么?",
    ]
    
    constraint = "如果你不确定或不知道,请直接说'我不知道',不要编造。"

    async with asyncio.TaskGroup() as tg:
        tasks_dict: dict[str, list] = {}
        for p in prompts:
            # 不带约束提示词
            task1 = tg.create_task(asyncio.to_thread(chat, input=p, temperature=0.0))
            # 带约束提示词
            task2 = tg.create_task(asyncio.to_thread(chat, input=f"{p},{constraint}", temperature=0.0))
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