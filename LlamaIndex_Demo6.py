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

# -------------------------- 混合检索+智谱重排核心函数（修复QueryBundle序列化问题） --------------------------
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=3)

def hybrid_retrieve_and_rerank(query_bundle):
    # ✅ 核心修复：提取QueryBundle中的纯字符串，而不是传整个对象
    query_str = query_bundle.query_str
    print(f"\n🔍 原始查询字符串：{query_str}")
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
    
    # 步骤2：智谱官方重排API调用（仅当有检索结果时执行）
    print(f"🔍 混合检索得到 {len(unique_nodes)} 条候选节点，准备重排...")
    if len(unique_nodes) == 0:
        return []
    
    # 提取文档内容
    docs = [node.node.get_content() for node in unique_nodes]
    
    # 调用智谱重排API（参数格式严格对齐官方要求）
    url = "https://ai.gitee.com/v1/rerank"
    headers = {
        "Authorization": f"Bearer 6HOQLKHJXMCAV7AT4V3CFXOXYVCMMPMKPPS2KXM2",
        "Content-Type": "application/json"
    }
    data = {
        "model": "bce-reranker-base_v1",
        "query": query_str,  # ✅ 传纯字符串，不是QueryBundle对象
        "documents": docs
    }
    
    # 发送请求并处理响应
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=30)
        resp.raise_for_status()  # 抛出HTTP错误
        resp_json = resp.json()
        print(f"✅ 重排API调用成功，返回结果：{resp_json}")
    except Exception as e:
        print(f"⚠️ 重排API调用失败，使用原始检索结果：{str(e)}")
        return unique_nodes[:2]  # 降级返回前2条
    
    # 按重排结果重新排序节点
    reranked_nodes = []
    for item in resp_json.get("results", []):
        idx = item.get("index", 0)
        if 0 <= idx < len(unique_nodes):
            reranked_nodes.append(
                NodeWithScore(
                    node=unique_nodes[idx].node,
                    score=item.get("relevance_score", 0.0)
                )
            )
    
    return reranked_nodes

# -------------------------- 最终查询引擎 --------------------------
query_engine = RetrieverQueryEngine(
    retriever=type('obj', (object,), {'retrieve': hybrid_retrieve_and_rerank})
)

# 测试运行
try:
    response = query_engine.query("文档里的核心内容是什么？")
    print("\n🤖 优化后回答：", response)
except Exception as e:
    print(f"\n❌ 运行出错：{str(e)}")