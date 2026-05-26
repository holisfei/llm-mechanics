from llm_fetcher import fetch, MessageModel, ROLE


messages: list[MessageModel] = []

def chat_with_no_memory(input: str):
    params:dict = {
        "model": "glm-4.5-air",
        "messages": [
            {'role': 'user', 'content': input},
        ],
        "stream": False,
    }
    result = fetch(params)
    print(f"{result.content}\n{result.usage.output}")

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