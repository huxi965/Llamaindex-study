from pymilvus import MilvusClient, DataType
import time

# 1. 连接 Milvus（docker-compose 启动的服务）
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("✅ 已连接 Milvus 服务")

# 2. 定义集合名（后续操作都用这个集合）
COLLECTION_NAME = "milvus_learning"

# 3. 清空旧数据（方便重复测试）
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 4. 创建集合 Schema（核心：定义数据结构）
# Schema = 向量数据库的「表结构」，必须明确字段类型/维度
schema = client.create_schema(auto_id=True)  # auto_id=True：主键自动生成
# 主键字段（必选，唯一标识每条数据）
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True
)
# 文本字段（存储原始内容，可选）
schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,
    max_length=65535  # VARCHAR 必须指定最大长度
)
# 向量字段（核心！必须指定维度，智谱/OpenAI 一般是 1024/768 维）
schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=4  # 简化测试用 4 维，实际用 1024 维
)

# 5. 创建索引（核心！加速向量检索，无索引则全表扫描）
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",          # 要索引的向量字段
    index_type="HNSW",            # 工业级索引（RAG 最常用）
    metric_type="COSINE",         # 相似度计算方式（文本用余弦最优）
    params={"M": 8, "efConstruction": 64}  # 索引参数（默认值即可）
)

# 6. 正式创建集合
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)
print("✅ 集合创建成功")

# 7. 插入测试数据（3 条 4 维向量）
test_data = [
    {"content": "机器学习是人工智能的核心", "vector": [0.1, 0.2, 0.3, 0.4]},
    {"content": "大模型可以理解自然语言", "vector": [0.2, 0.3, 0.4, 0.5]},
    {"content": "向量数据库用于存储Embedding", "vector": [0.3, 0.4, 0.5, 0.6]}
]
insert_result = client.insert(
    collection_name=COLLECTION_NAME,
    data=test_data
)
print(f"✅ 插入 {len(insert_result)} 条数据，数据ID：{insert_result}")

# 8. 关键：插入后等待索引生效（HNSW 索引需要几秒构建）
time.sleep(10)

# 9. 向量检索（核心操作，修复越界问题）
query_vector = [0.1, 0.2, 0.3, 0.4]  # 完全匹配第一条数据
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],             # 检索向量（列表格式，支持多向量）
    limit=3,                         # 返回最多 3 条结果
    output_fields=["content"],       # 返回哪些字段（除了向量都可以指定）
    search_params={"ef": 64}         # 检索参数（和索引参数匹配）
)

# 安全处理检索结果（避免越界）
print("\n🎯 向量检索结果：")
if results and len(results[0]) > 0:
    for rank, res in enumerate(results[0], 1):
        # 余弦距离越小 → 相似度越高（1 - 距离 = 相似度百分比）
        similarity = 1 - res["distance"]
        print(f"第{rank}名 | 相似度：{similarity:.4f} | 内容：{res['entity']['content']}")
else:
    print("❌ 无匹配的检索结果")

# 10. 查看集合数据量（验证数据插入成功）
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"\n📊 集合总数据量：{stats['row_count']} 条")