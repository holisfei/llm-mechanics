from dotenv import load_dotenv
from zai import ZhipuAiClient
import zai

# 提前加载环境变量
load_dotenv()

client = ZhipuAiClient()


# 模型是一个 token 一个 token 生成的,
# 流式输出能让我们看到这个过程
def stream_chat(promt: str):
    # chunk 计数器
    chunk_count: int = 0

    try:
        response = client.chat.completions.create(
            model="glm-4.5-air",
            messages=[
                # {'role': 'system', 'content': '你是一个百科知识助手'},
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
                # 不能用return，return会直接结束掉函数
                # yield 生成器，交给外部去遍历 
    except zai.core.APIStatusError as err:
        return f"API 状态错误: {err}"
    except zai.core.APITimeoutError as err:
        return f"请求超时: {err}"
    except Exception as err:
        return f"其他错误: {err}" 
        
def main():
    res = stream_chat("为什么会发生地震？")
    for r in res:
        print(r, end='', flush=True)
    
if __name__ == "__main__":
    main()



