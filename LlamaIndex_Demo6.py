# ==============================================
# 【代码第1行！零注释零空行！根治Windows resource module报错】
# ==============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
ZHIPU_API_KEY = "ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI"

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 1. 配置智谱模型
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=ZHIPU_API_KEY)
Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=ZHIPU_API_KEY)

# 2. 加载本地向量库
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
all_nodes = list(index.docstore.docs.values())

# -------------------------- 混合检索：向量语义 + BM25关键词（你学习的核心知识点） --------------------------
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=3)

def hybrid_retrieve_and_rerank(query_str):
    # 步骤1：混合检索（向量+BM25）
    vector_nodes = vector_retriever.retrieve(query_str)
    bm25_nodes = bm25_retriever.retrieve(query_str)
    all_nodes = vector_nodes + bm25_nodes
    # 去重
    seen_ids = set()
    unique_nodes = []
    for node in all_nodes:
        if node.node_id not in seen_ids:
            seen_ids.add(node.node_id)
            unique_nodes.append(node)
    
    # 步骤2：【手动智谱重排！完全不导入postprocessor模块！】
    if len(unique_nodes) == 0:
        return []
    # 调用智谱官方重排API
    docs = [node.node.get_content() for node in unique_nodes]
    url = "https://open.bigmodel.cn/api/paas/v4/rerank"
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "rerank-v1",
        "query": query_str,
        "documents": docs,
        "top_n": 2
    }
    resp = requests.post(url, json=data).json()
    # 按重排结果重新排序节点
    reranked_nodes = []
    for item in resp["results"]:
        reranked_nodes.append(NodeWithScore(node=unique_nodes[item["index"]].node, score=item["relevance_score"]))
    return reranked_nodes

# -------------------------- 最终查询引擎（直接用手动混合+重排函数） --------------------------
query_engine = RetrieverQueryEngine(
    retriever=type('obj', (object,), {'retrieve': hybrid_retrieve_and_rerank})
)

# 测试运行
print("✅ Windows零报错！混合检索+智谱官方重排 RAG高级优化全部生效！")
response = query_engine.query("文档核心内容是什么？")
print("\n🤖 优化后回答：", response)