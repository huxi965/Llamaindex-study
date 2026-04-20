import os
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 1. 配置模型
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY, temperature = 0.7)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# 2. 加载本地向量库
storage_context = StorageContext.from_defaults(persist_dir="./storage_memory")
index = load_index_from_storage(storage_context)

# 核心：创建对话引擎（自带记忆！）
chat_engine = index.as_chat_engine(
    chat_mode="context",  # 上下文对话模式
    verbose=True,          # 打印检索过程（方便调试）
    similarity_top_k=2     # 每轮检索3条相关内容（更精准）
)

# 3. 多轮对话测试
print("🤖 文档聊天机器人已启动（输入 exit 退出）！")
while True:
    user_input = input("你：")
    if user_input.lower() == "exit":
        print("🤖 再见！")
        break
    response = chat_engine.chat(user_input)
    print("机器人：", response)