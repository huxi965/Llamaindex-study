from pymilvus import MilvusClient, DataType
import time
from zhipuai import ZhipuAI

# ===================== 1. 初始化工具 =====================
# 1.1 连接 Milvus（docker-compose 启动的服务，地址不变）
milvus_client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
COLLECTION_NAME = "milvus_rag_demo"  # 实战用集合名（语义检索）

# 1.2 初始化智谱 Embedding 客户端（替换成你的 API Key）
zhipu_client = ZhipuAI(api_key="ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI")

# ===================== 2. 工具函数：文本转 Embedding 向量 =====================
# 核心函数：把任意文本转成 1024 维向量（智谱 Embedding-2 模型）
def text_to_embedding(text: str) -> list:
    """
    文本转 Embedding 向量（适配 Milvus 存储）
    :param text: 要转换的文本
    :return: 1024 维 float 向量
    """
    try:
        response = zhipu_client.embeddings.create(
            model="embedding-2",  # 智谱Embedding模型，输出1024维向量
            input=text
        )
        # 提取向量并转成float（Milvus要求float类型）
        vector = response.data[0].embedding
        return [float(x) for x in vector]
    except Exception as e:
        print(f"❌ 文本转向量失败：{e}")
        return []

# ===================== 3. 重建 Milvus 集合（适配 1024 维真实向量） =====================
# 清空旧集合（方便重复测试）
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

# 定义 Schema（核心：向量维度改为 1024，适配真实 Embedding）
schema = milvus_client.create_schema(auto_id=True)
# 主键字段
schema.add_field("id", DataType.INT64, is_primary=True)
# 原始文本字段（存储待检索的文本）
schema.add_field("text", DataType.VARCHAR, max_length=65535)
# 向量字段（1024 维，和智谱模型输出一致）
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# 创建 HNSW 索引（适配 1024 维向量的最优参数）
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",  # 文本语义检索必选余弦相似度
    params={"M": 16, "efConstruction": 128}  # 1024维向量推荐参数
)

# 正式创建集合
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)
print("✅ Milvus 集合创建成功（适配 1024 维真实 Embedding）")

# ===================== 4. 真实文本入库（向量化 + 插入 Milvus） =====================
# 待入库的文本（模拟知识库，比如AI相关文档）
knowledge_texts = [
    "Milvus 是一款开源的分布式向量数据库，专为海量高维向量的高效检索设计",
    "向量检索是大模型 RAG 架构的核心环节，决定了检索精度和响应速度",
    "HNSW 索引是 Milvus 中性能最优的向量索引，适合文本语义检索场景",
    "智谱 Embedding-2 模型能将文本转换为 1024 维的稠密向量，适配 Milvus 存储",
    "RAG（检索增强生成）的流程是：检索相关知识 → 拼接Prompt → 大模型生成回答"
]

# 批量向量化 + 插入 Milvus
insert_data = []
for text in knowledge_texts:
    vector = text_to_embedding(text)
    if vector:  # 确保向量生成成功
        insert_data.append({
            "text": text,
            "embedding": vector
        })

# 插入 Milvus
if insert_data:
    milvus_client.insert(COLLECTION_NAME, data=insert_data)
    milvus_client.flush(COLLECTION_NAME)
    time.sleep(5)  # 1024维向量索引构建稍慢，等待5秒
    print(f"✅ 成功插入 {len(insert_data)} 条真实文本（已转1024维向量）")
    print(f"📊 集合总数据量：{milvus_client.get_collection_stats(COLLECTION_NAME)['row_count']} 条")
else:
    print("❌ 无有效数据插入，请检查Embedding API Key是否正确")

# ===================== 5. 真实语义检索（输入问题，找最相关的文本） =====================
# 待检索的问题（自然语言，模拟用户提问）
query_question = "Milvus 中哪种索引适合文本语义检索？"
print(f"\n🔍 用户查询：{query_question}")
# 步骤1：问题转 Embedding 向量
query_vector = text_to_embedding(query_question)
if not query_vector:
    exit()

# 步骤2：Milvus 语义检索（找最相关的3条文本）
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=10,
    output_fields=["text"],
    search_params={"ef": 128}  # 检索参数和索引参数匹配
)

# 步骤3：整理检索结果（按相似度排序）
print("\n🎯 语义检索结果（按相关性排序）：")
if search_results and len(search_results[0]) > 0:
    for rank, res in enumerate(search_results[0], 1):
        similarity = 1 - res["distance"]  # 余弦距离转相似度
        print(f"\n第{rank}名 | 相似度：{similarity:.4f}")
        print(f"文本内容：{res['entity']['text']}")
else:
    print("❌ 未检索到相关内容")