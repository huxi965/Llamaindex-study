import os
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext
)
# 导入分块器
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 1. 配置模型
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# 核心：自定义分块参数（自己随便改！）
Settings.node_parser = SentenceSplitter(
    chunk_size=512,       # 每个片段512字符（小片段更精准，大片段信息全）
    chunk_overlap=50,     # 重叠50字符（防止一句话被切断）
    separator="\n"        # 按换行符分割（更符合文档格式）
)

# 2. 加载文档
documents = SimpleDirectoryReader(
    input_dir="docs",
    required_exts=[".pdf", ".txt", ".md"]
).load_data()

# 3. 构建索引 + 持久化保存（覆盖旧向量库）
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")
print("✅ 自定义分块完成，向量库已保存！")

# 4. 测试查询
query_engine = index.as_query_engine()
response = query_engine.query("文档里的核心内容是什么？")
print("\n🤖 回答：", response)