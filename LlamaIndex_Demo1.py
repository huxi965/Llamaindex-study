import os

# 智谱API Key
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"

# 导入组件
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 配置智谱模型
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)


# ===================== 重点修改：加载 docs 文件夹 =====================
documents = SimpleDirectoryReader(input_dir="docs", recursive=True).load_data()
# ====================================================================

# 构建索引+查询
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 提问
response = query_engine.query("文档里讲了什么内容？")
print(response)