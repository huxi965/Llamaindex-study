# 第1行！关闭Windows警告
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# -------------------------- 配置 --------------------------
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# -------------------------- 读取文档（必须有） --------------------------
documents = SimpleDirectoryReader("./docs").load_data()

# -------------------------- Chroma 向量库 --------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="my_rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -------------------------- 创建索引 --------------------------
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# -------------------------- 最简单查询引擎（只使用向量检索！无BM25） --------------------------
query_engine = index.as_query_engine()

# -------------------------- 运行 --------------------------
print("✅ Chroma 运行成功！开始学习！")
resp = query_engine.query("文档讲了什么？")
print(resp)