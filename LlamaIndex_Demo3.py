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

# 配置模型（必须保留，加载索引需要）
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# ===================== 1. 加载本地保存的向量库 =====================
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ===================== 2. 核心：遍历并打印向量库所有内容 =====================
print("="*50)
print("📂 开始查看向量库中存储的所有内容")
print("="*50)

# 获取向量库里所有的节点（原始数据+向量都在这里）
all_nodes = index.docstore.docs.values()

for i, node in enumerate(all_nodes):
    print(f"\n🔹 第 {i+1} 条存储内容")
    print("-"*30)
    # 1. 查看【原始文本数据】（你文档里的真实内容）
    print(f"📝 原始文本：\n{node.text}")
    print("-"*30)
    # 2. 查看【向量数据】（Embedding生成的数字，只打印前10位，避免太长）
  #  print(f"🔢 向量数据（前10位）：{node.embedding[:10]}...")
    print("-"*30)
    # 3. 查看【元数据】（文件来源、页码等信息）
    print(f"📄 元数据：{node.metadata}")
    print("="*50)