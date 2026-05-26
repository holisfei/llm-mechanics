from dotenv import load_dotenv
import zai
from zai import ZhipuAiClient
import pandas as pd
import asyncio

load_dotenv()
client = ZhipuAiClient()
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.max_colwidth', None)  # 单元格内容不截断
pd.set_option('display.width', None)         # 自动适应终端宽度

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

    if is_async: # 并发执行
        async with asyncio.TaskGroup() as tg: # asyncio.to_thread 将一个同步函数加入异步任务
            tasks = [tg.create_task(asyncio.to_thread(fetch, kwargs)) for _ in range(count)]
        return [task.result() for task in tasks]
    else: # 同步执行
        results = []
        for i in range(count):
            result = fetch(kwargs)
            results.append(result)
        return results

async def run_params(
    temperature_list: list[float] | None = None, 
    top_p_list: list[float] | None = None, 
    do_sample: bool | None = None,
    prompt: str = "中国的首都是"
):
    # temperature
    results: dict[float, list[str]] = {}

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