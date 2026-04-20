import os
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 配置模型
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# 1. 加载文档
documents = SimpleDirectoryReader(
    input_dir="docs",
    recursive=True,
    required_exts=[".pdf", ".txt", ".md"]
).load_data()

# 2. 创建索引 + 保存到本地磁盘（storage文件夹）
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")  # 关键：保存索引

# 3. 查询
query_engine = index.as_query_engine()
response = query_engine.query("文档里讲了什么内容？")
print(response)